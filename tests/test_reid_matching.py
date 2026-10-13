"""
tests/test_reid_matching.py
IdentityResolver: the numpy-prefiltered matcher must make exactly the decisions the original
one-departure-at-a-time loop made, stay bounded and thread-safe, and survive departures that share
gid / camera / class (the generated dataclass ``__eq__`` compared numpy arrays and raised).
"""

import random
import threading
import time
import unittest
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from backend.reid import Departure, IdentityResolver
from backend.topology_engine import TopologyEngine


# ── the implementation this replaced, kept verbatim as the oracle ────────────────

@dataclass(eq=False)
class _RefDeparture:
    gid: int
    camera_id: int
    class_name: str
    embedding: np.ndarray
    t_depart: float


class ReferenceResolver:
    def __init__(self, topology=None, start_id=1, same_camera_threshold=0.93,
                 cross_camera_threshold=0.88, ttl_s=900.0, max_pending=1500):
        self.topology = topology
        self.same_camera_threshold = same_camera_threshold
        self.cross_camera_threshold = cross_camera_threshold
        self.ttl_s = ttl_s
        self._next_id = max(1, int(start_id))
        self._pending: Deque[_RefDeparture] = deque(maxlen=max_pending)
        self._window_cache: Dict[Tuple[int, int], Optional[Tuple[float, float]]] = {}

    def register_departure(self, gid, camera_id, class_name, embedding, t_depart):
        self._pending.append(_RefDeparture(gid, camera_id, class_name, embedding, t_depart))

    def _window(self, src, dst):
        key = (src, dst)
        if key not in self._window_cache:
            window = None
            if self.topology is not None:
                try:
                    t_min, t_max = self.topology.calculate_arrival_window(src, dst)
                    if t_max != float("inf"):
                        window = (float(t_min), float(t_max))
                except KeyError:
                    window = None
            self._window_cache[key] = window
        return self._window_cache[key]

    def _transit_score(self, dep, camera_id, elapsed):
        if dep.camera_id == camera_id:
            return 1.0 if 0.0 <= elapsed <= 30.0 else 0.0
        window = self._window(dep.camera_id, camera_id)
        if window is None:
            return 0.0
        valid, score = self.topology.score_transition_window(window[0], window[1], elapsed)
        return score if valid else 0.0

    def resolve(self, camera_id, class_name, embedding, timestamp):
        while self._pending and timestamp - self._pending[0].t_depart > self.ttl_s:
            self._pending.popleft()
        best, best_score = None, 0.0
        for dep in self._pending:
            if dep.class_name != class_name:
                continue
            elapsed = timestamp - dep.t_depart
            if elapsed < 0:
                continue
            transit = self._transit_score(dep, camera_id, elapsed)
            if transit <= 0.0:
                continue
            sim = float(np.dot(embedding, dep.embedding))
            threshold = (self.same_camera_threshold if dep.camera_id == camera_id
                         else self.cross_camera_threshold)
            if sim < threshold:
                continue
            score = 0.7 * sim + 0.3 * transit
            if score > best_score:
                best, best_score = dep, score
        if best is not None:
            self._pending.remove(best)
            return best.gid, round(best_score, 4), best.camera_id
        gid = self._next_id
        self._next_id += 1
        return gid, 0.0, None


# ── test fixtures ────────────────────────────────────────────────────────────────

class RandomTopology:
    """Random directed arrival windows between ``n`` cameras; some pairs unknown or unreachable."""

    score_transition_window = staticmethod(TopologyEngine.score_transition_window)

    def __init__(self, n_cameras, rng):
        self.ids = list(range(100, 100 + n_cameras))
        self.table = {}
        for a in self.ids:
            for b in self.ids:
                if a == b:
                    continue
                r = rng.random()
                if r < 0.25:
                    continue                                   # no such pair -> KeyError
                if r < 0.35:
                    self.table[(a, b)] = (float("inf"), float("inf"))
                    continue
                t_min = rng.choice([0.0, rng.uniform(0, 20), rng.uniform(20, 120)])
                self.table[(a, b)] = (t_min, t_min + rng.uniform(0, 240))

    def calculate_arrival_window(self, src, dst):
        if (src, dst) not in self.table:
            raise KeyError((src, dst))
        return self.table[(src, dst)]


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def with_cosine(rng, base, cos):
    """A unit vector whose float32 dot with ``base`` is ``cos`` (to float32 precision)."""
    other = rng.standard_normal(base.shape[0])
    other -= other.dot(base) * base
    other /= np.linalg.norm(other)
    return unit(cos * base.astype(np.float64) + np.sqrt(1.0 - cos * cos) * other)


class RandomWorkload:
    """A reproducible mix of departures and queries whose similarities straddle the thresholds."""

    def __init__(self, seed, dim=32, n_cameras=8, classes=("car", "motorcycle", "truck")):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.topology = RandomTopology(n_cameras, self.rng)
        self.classes = classes
        self.dim = dim
        self.protos = [unit(self.np_rng.standard_normal(dim)) for _ in range(5)]

    def embedding(self):
        proto = self.rng.choice(self.protos)
        r = self.rng.random()
        if r < 0.15:
            cos = self.rng.choice([0.93, 0.88, 0.9299999, 0.9300001, 0.8799999, 0.8800001])  # on the thresholds
        else:
            cos = self.rng.uniform(0.80, 1.0)
        emb = with_cosine(self.np_rng, proto, min(cos, 1.0 - 1e-9))
        if self.rng.random() < 0.1:
            emb = emb.astype(np.float64)                # callers may hand in float64
        return emb


def run_equivalence(test, seed, ops=500, max_pending=60, ttl_s=240.0, dim=32):
    w = RandomWorkload(seed, dim=dim)
    new = IdentityResolver(topology=w.topology, max_pending=max_pending, ttl_s=ttl_s)
    old = ReferenceResolver(topology=w.topology, max_pending=max_pending, ttl_s=ttl_s)
    now, gid, queries, matches = 1000.0, 1, 0, 0
    for step in range(ops):
        now += w.rng.expovariate(1 / 4.0)
        if w.rng.random() < 0.55:
            cam = w.rng.choice(w.topology.ids)
            cls = w.rng.choice(w.classes)
            emb = w.embedding()
            t_dep = now - w.rng.choice([0.0, 0.0, w.rng.uniform(0, 30), w.rng.uniform(0, 90)])  # slightly out of order
            if w.rng.random() < 0.1:
                t_dep = now + w.rng.uniform(0, 5)        # "departed" in the future: never matches yet
            new.register_departure(gid, cam, cls, emb, t_dep)
            old.register_departure(gid, cam, cls, emb, t_dep)
            if w.rng.random() < 0.15:                    # an exact duplicate (score ties -> oldest wins)
                new.register_departure(gid + 1, cam, cls, emb, t_dep)
                old.register_departure(gid + 1, cam, cls, emb, t_dep)
                gid += 1
            gid += 1
        else:
            cam = w.rng.choice(w.topology.ids)
            cls = w.rng.choice(w.classes)
            emb = w.embedding()
            got, want = new.resolve(cam, cls, emb, now), old.resolve(cam, cls, emb, now)
            test.assertEqual(got, want, f"seed {seed} step {step}")
            queries += 1
            matches += want[2] is not None
        test.assertEqual(new.pending_count(), len(old._pending), f"seed {seed} step {step}")
        if step % 25 == 0:
            test.assertEqual([d.gid for d in new.pending_snapshot()], [d.gid for d in old._pending])
    test.assertEqual([d.gid for d in new.pending_snapshot()], [d.gid for d in old._pending])
    return queries, matches


class MatchingEquivalenceTest(unittest.TestCase):
    def test_decisions_match_the_original_loop_on_randomised_workloads(self):
        total_q = total_m = 0
        for seed in range(12):
            q, m = run_equivalence(self, seed)
            total_q, total_m = total_q + q, total_m + m
        self.assertGreaterEqual(total_q, 1500)
        self.assertGreater(total_m, 150, "workload should produce plenty of real matches, not just new ids")

    def test_equivalence_with_real_embedding_size_a_full_ring_and_ttl_expiry(self):
        q, m = run_equivalence(self, seed=99, ops=900, max_pending=1500 // 10, ttl_s=120.0, dim=256)
        self.assertGreater(q, 300)
        self.assertGreater(m, 10)

    def test_tiny_rings_wrap_and_compact_correctly(self):
        for cap in (1, 2, 3, 7):
            run_equivalence(self, seed=cap, ops=300, max_pending=cap)

    def test_without_a_topology_only_same_camera_reappearances_match(self):
        new = IdentityResolver(topology=None)
        old = ReferenceResolver(topology=None)
        rng = np.random.default_rng(3)
        emb = unit(rng.standard_normal(16))
        for r in (new, old):
            r.register_departure(1, 5, "car", emb, 100.0)
            r.register_departure(2, 6, "car", emb, 100.0)
        for cam, t in ((9, 105.0), (6, 140.0), (6, 110.0)):
            self.assertEqual(new.resolve(cam, "car", emb, t), old.resolve(cam, "car", emb, t))

    def test_topology_with_its_own_scoring_function_is_not_prefiltered_by_the_gaussian_bound(self):
        class Lenient(RandomTopology):
            @staticmethod
            def score_transition_window(t_min, t_max, t):     # accepts anything within 10 min
                return (True, 0.5) if abs(t - t_min) < 600 else (False, 0.0)
        rng = random.Random(4)
        topo = Lenient(4, rng)
        topo.table = {(100, 101): (10.0, 20.0)}
        emb = unit(np.random.default_rng(1).standard_normal(16))
        new, old = IdentityResolver(topology=topo), ReferenceResolver(topology=topo)
        for r in (new, old):
            r.register_departure(7, 100, "car", emb, 0.0)
        self.assertEqual(new.resolve(101, "car", emb, 400.0), old.resolve(101, "car", emb, 400.0))
        self.assertEqual(new.pending_count(), 0)

    def test_nan_and_zero_embeddings_behave_like_the_original(self):
        w = RandomWorkload(5)
        cams = w.topology.ids[:2]
        for query in (np.full(32, np.nan, dtype=np.float32), np.zeros(32, dtype=np.float32)):
            new, old = IdentityResolver(topology=w.topology), ReferenceResolver(topology=w.topology)
            for r in (new, old):
                r.register_departure(1, cams[0], "car", w.protos[0], 0.0)
            self.assertEqual(new.resolve(cams[0], "car", query, 5.0), old.resolve(cams[0], "car", query, 5.0))


class PendingStructureTest(unittest.TestCase):
    def setUp(self):
        self.emb = unit(np.random.default_rng(0).standard_normal(16))

    def test_two_departures_sharing_gid_camera_and_class_do_not_break_matching(self):
        # Regression: Departure was a dataclass with eq=True; removing the matched one compared it to
        # an earlier look-alike, which compared the numpy embeddings and raised ValueError.
        r = IdentityResolver()
        far = with_cosine(np.random.default_rng(1), self.emb, 0.5)
        r.register_departure(7, 3, "car", far, 100.0)         # same gid / camera / class, other look
        r.register_departure(7, 3, "car", self.emb, 100.0)
        gid, conf, cam = r.resolve(3, "car", self.emb, 110.0)
        self.assertEqual((gid, cam), (7, 3))
        self.assertEqual(r.pending_count(), 1, "only the matched departure is consumed")
        self.assertIs(r.pending_snapshot()[0].embedding, far)

    def test_departure_has_identity_equality(self):
        a = Departure(1, 2, "car", self.emb, 3.0)
        b = Departure(1, 2, "car", self.emb.copy(), 3.0)
        self.assertNotEqual(a, b)
        self.assertEqual(a, a)
        self.assertEqual([a, b].index(b), 1)

    def test_pending_is_bounded_and_evicts_the_oldest(self):
        r = IdentityResolver(max_pending=5)
        for gid in range(1, 40):
            r.register_departure(gid, 1, "car", self.emb, 100.0 + gid)
            self.assertLessEqual(r.pending_count(), 5)
        self.assertEqual([d.gid for d in r.pending_snapshot()], [35, 36, 37, 38, 39])

    def test_consumed_departures_leave_room_without_evicting_live_ones(self):
        r = IdentityResolver(max_pending=4)
        for gid in range(1, 5):
            r.register_departure(gid, 1, "car", self.emb, 100.0)
        self.assertEqual(r.resolve(1, "car", self.emb, 101.0)[0], 1)    # oldest of equal scores
        self.assertEqual(r.resolve(1, "car", self.emb, 101.0)[0], 2)
        for gid in range(10, 16):
            r.register_departure(gid, 1, "car", self.emb, 100.0)
        self.assertEqual([d.gid for d in r.pending_snapshot()], [12, 13, 14, 15])

    def test_ttl_expires_departures_from_the_front(self):
        r = IdentityResolver(ttl_s=60.0)
        r.register_departure(1, 1, "car", self.emb, 0.0)
        r.register_departure(2, 1, "car", self.emb, 100.0)
        r.resolve(1, "truck", self.emb, 100.0)
        self.assertEqual([d.gid for d in r.pending_snapshot()], [2])

    def test_wrong_sized_embedding_is_rejected_at_registration(self):
        r = IdentityResolver()
        r.register_departure(1, 1, "car", self.emb, 0.0)
        with self.assertRaises(ValueError):
            r.register_departure(2, 1, "car", np.zeros(7, dtype=np.float32), 0.0)
        self.assertEqual(r.pending_count(), 1)

    def test_concurrent_registration_and_resolution_stay_consistent(self):
        w = RandomWorkload(11)
        r = IdentityResolver(topology=w.topology, max_pending=200)
        errors, stop = [], threading.Event()
        embs = [w.embedding() for _ in range(40)]

        def producer(offset):
            try:
                for i in range(1500):
                    r.register_departure(offset + i, w.topology.ids[i % 8], "car", embs[i % 40], 1000.0 + i * 0.01)
            except Exception as exc:       # noqa: BLE001
                errors.append(exc)

        def consumer():
            try:
                i = 0
                while not stop.is_set():
                    r.resolve(w.topology.ids[i % 8], "car", embs[i % 40], 1000.0 + i * 0.01)
                    i += 1
                    self_count = r.pending_count()
                    if self_count > 200:
                        errors.append(AssertionError(f"pending grew to {self_count}"))
            except Exception as exc:       # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=producer, args=(k * 10000,)) for k in range(3)]
        reader = threading.Thread(target=consumer)
        reader.start()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stop.set()
        reader.join()
        self.assertEqual(errors, [])
        self.assertLessEqual(r.pending_count(), 200)
        self.assertEqual(len(r.pending_snapshot()), r.pending_count())


class MatchingSpeedTest(unittest.TestCase):
    def test_full_ring_query_is_faster_than_the_original_loop(self):
        w = RandomWorkload(21, dim=256, n_cameras=40)
        new = IdentityResolver(topology=w.topology, max_pending=1500)
        old = ReferenceResolver(topology=w.topology, max_pending=1500)
        for i in range(1500):
            emb = w.embedding()
            cam = w.topology.ids[i % 40]
            t = 1000.0 + i * 0.4
            new.register_departure(i, cam, "car", emb, t)
            old.register_departure(i, cam, "car", emb, t)
        queries = [(w.topology.ids[k % 40], w.embedding()) for k in range(120)]
        now = 1000.0 + 1500 * 0.4

        def bench(fn):
            t0 = time.perf_counter()
            for cam, emb in queries:
                fn(cam, emb)
            return (time.perf_counter() - t0) / len(queries)

        # Non-consuming comparison: identical resolvers, queries that cannot match anything.
        miss = [(cam, np.zeros(256, dtype=np.float32)) for cam, _ in queries]
        queries_backup, queries[:] = list(queries), miss
        t_new = min(bench(lambda c, e: new.resolve(c, "car", e, now)) for _ in range(3))
        t_old = min(bench(lambda c, e: old.resolve(c, "car", e, now)) for _ in range(3))
        queries[:] = queries_backup
        self.assertLess(t_new, t_old, f"new {t_new * 1e3:.3f} ms vs old {t_old * 1e3:.3f} ms per call")


if __name__ == "__main__":
    unittest.main()
