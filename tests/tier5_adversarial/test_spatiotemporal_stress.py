#!/usr/bin/env python3
"""
Tier 5 Adversarial Stress Tests: Spatiotemporal Feasibility Model & Arrival Windows
BSW Kota Bogor Multi-Camera Tracking ("Follow That Car") Traffic Intelligence Dashboard

Verifies:
- All 220 edges satisfy t_min > 0 and t_max > t_min strictly.
- Boundary values: t = t_min - eps, t = t_min, midpoint, t = t_max, t = t_max + eps.
- Score S_st in [0.0, 1.0] behaves monotonically and smoothly decays outside [t_min, t_max].
- Negative elapsed times, non-finite values, and impossible transitions return (False, 0.0).
- Multi-hop arrival window calculations across all 8,836 node pairs with 100% reachability.
- High-throughput sub-millisecond query latency across 10,000 queries.
"""

import unittest
import math
import random
import time
from tests import TOTAL_CAMERAS, TOTAL_EDGES
from backend.topology_engine import (
    TopologyEngine,
    is_valid_transition,
    calculate_arrival_window
)


class TestSpatiotemporalAdversarial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = TopologyEngine()
        cls.cameras = cls.engine.get_all_cameras()
        cls.edges = cls.engine.get_all_edges()
        cls.cam_ids = sorted([c["id"] for c in cls.cameras])

    def test_all_220_edges_arrival_window_invariants(self):
        """Verify t_min > 0, t_max > t_min, and distance > 0 across all 220 edges."""
        self.assertEqual(len(self.edges), TOTAL_EDGES)
        for edge in self.edges:
            u = edge["source_id"]
            v = edge["target_id"]
            t_min = edge["t_min_s"]
            t_max = edge["t_max_s"]
            dist = edge["road_distance_m"]

            self.assertGreater(t_min, 0.0, f"Edge ({u}->{v}) has non-positive t_min: {t_min}")
            self.assertGreater(t_max, t_min, f"Edge ({u}->{v}) t_max ({t_max}) <= t_min ({t_min})")
            self.assertGreater(dist, 0.0, f"Edge ({u}->{v}) has non-positive distance: {dist}")

            calc_min, calc_max = self.engine.calculate_arrival_window(u, v)
            self.assertGreater(calc_min, 0.0)
            self.assertGreater(calc_max, calc_min)

    def test_boundary_values_and_smooth_decay(self):
        """Stress-test exact boundaries, epsilon offsets, and Gaussian decay behavior on all edges."""
        for edge in self.edges:
            u = edge["source_id"]
            v = edge["target_id"]
            t_min, t_max = self.engine.calculate_arrival_window(u, v)
            t_mid = (t_min + t_max) / 2.0

            # 1. Exact t_min
            val_min, s_min = self.engine.is_valid_transition(u, v, t_min)
            self.assertTrue(val_min)
            self.assertEqual(s_min, 1.0)

            # 2. Midpoint
            val_mid, s_mid = self.engine.is_valid_transition(u, v, t_mid)
            self.assertTrue(val_mid)
            self.assertEqual(s_mid, 1.0)

            # 3. Exact t_max
            val_max, s_max = self.engine.is_valid_transition(u, v, t_max)
            self.assertTrue(val_max)
            self.assertEqual(s_max, 1.0)

            # 4. Infinitesimally early (t = t_min - 1e-4) -> valid and ~1.0
            val_e_eps, s_e_eps = self.engine.is_valid_transition(u, v, t_min - 0.0001)
            self.assertTrue(val_e_eps)
            self.assertGreaterEqual(s_e_eps, 0.999)
            self.assertLessEqual(s_e_eps, 1.0)

            # 5. Infinitesimally late (t = t_max + 1e-4) -> valid and ~1.0
            val_l_eps, s_l_eps = self.engine.is_valid_transition(u, v, t_max + 0.0001)
            self.assertTrue(val_l_eps)
            self.assertGreaterEqual(s_l_eps, 0.999)
            self.assertLessEqual(s_l_eps, 1.0)

            # 6. Moderate early decay (0.5 * sigma)
            sigma_early = max(0.25 * t_min, 2.0)
            t_early_half = max(0.0, t_min - 0.5 * sigma_early)
            if t_early_half > 0:
                val_eh, s_eh = self.engine.is_valid_transition(u, v, t_early_half)
                self.assertTrue(val_eh)
                self.assertGreaterEqual(s_eh, 0.80)
                self.assertLessEqual(s_eh, 0.95)

            # 7. Severe early (teleportation: 3.5 * sigma) -> rejected (False, 0.0)
            t_teleport = t_min - 3.5 * sigma_early
            if t_teleport >= 0:
                val_tele, s_tele = self.engine.is_valid_transition(u, v, t_teleport)
                self.assertFalse(val_tele)
                self.assertEqual(s_tele, 0.0)

            # 8. Severe late (3.5 * sigma) -> rejected (False, 0.0)
            sigma_late = max(0.50 * (t_max - t_min), 30.0)
            val_late_huge, s_late_huge = self.engine.is_valid_transition(u, v, t_max + 3.5 * sigma_late)
            self.assertFalse(val_late_huge)
            self.assertEqual(s_late_huge, 0.0)

    def test_monotonicity_decay_curve_sweep(self):
        """Verify mathematical monotonicity of S_st: increasing before t_min, constant in window, decreasing after t_max."""
        sampled_edges = self.edges[::5]  # Sample 44 representative edges
        for edge in sampled_edges:
            u, v = edge["source_id"], edge["target_id"]
            t_min, t_max = self.engine.calculate_arrival_window(u, v)

            # Early phase: 0 -> t_min (monotonic non-decreasing)
            prev_score = 0.0
            for i in range(51):
                t_eval = (i / 50.0) * t_min
                _, score = self.engine.is_valid_transition(u, v, t_eval)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)
                self.assertGreaterEqual(score, prev_score - 1e-4, f"Early monotonicity broken for ({u}->{v}) at {t_eval}")
                prev_score = score

            # Window phase: t_min -> t_max (strictly 1.0)
            for i in range(21):
                t_eval = t_min + (i / 20.0) * (t_max - t_min)
                val, score = self.engine.is_valid_transition(u, v, t_eval)
                self.assertTrue(val)
                self.assertEqual(score, 1.0)

            # Late phase: t_max -> t_max + 600s (monotonic non-increasing)
            prev_score = 1.0
            for i in range(51):
                t_eval = t_max + (i / 50.0) * 600.0
                _, score = self.engine.is_valid_transition(u, v, t_eval)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)
                self.assertLessEqual(score, prev_score + 1e-4, f"Late monotonicity broken for ({u}->{v}) at {t_eval}")
                prev_score = score

    def test_adversarial_impossible_and_negative_times(self):
        """Verify strict rejection of negative elapsed times, NaN, Inf, and impossible transitions."""
        test_edge = self.edges[0]
        u, v = test_edge["source_id"], test_edge["target_id"]

        # Negative elapsed times
        for neg_t in [-1e-9, -0.01, -1.0, -100.0, float("-inf")]:
            self.assertEqual(self.engine.is_valid_transition(u, v, neg_t), (False, 0.0))
            self.assertEqual(is_valid_transition(u, v, neg_t), (False, 0.0))

        # Non-finite values
        self.assertEqual(self.engine.is_valid_transition(u, v, float("nan")), (False, 0.0))
        self.assertEqual(self.engine.is_valid_transition(u, v, float("inf")), (False, 0.0))

        # Forbidden one-way transitions
        forbidden_ssa = [(87, 199), (83, 87), (81, 83), (85, 81), (133, 85), (79, 132)]
        for fu, fv in forbidden_ssa:
            self.assertEqual(self.engine.is_valid_transition(fu, fv, 20.0, direct_only=True), (False, 0.0))
            # Full loop is > 300s, so 20s must be (False, 0.0)
            self.assertEqual(self.engine.is_valid_transition(fu, fv, 20.0, direct_only=False), (False, 0.0))

        # Same camera tests
        self.assertEqual(self.engine.is_valid_transition(u, u, 0.0), (True, 1.0))
        self.assertEqual(self.engine.is_valid_transition(u, u, 30.0), (True, 1.0))
        val_lingering, score_lingering = self.engine.is_valid_transition(u, u, 60.0)
        self.assertTrue(val_lingering)
        self.assertAlmostEqual(score_lingering, 0.8825, places=3)
        self.assertEqual(self.engine.is_valid_transition(u, u, 7200.0), (False, 0.0))

    def test_multihop_all_pairs_arrival_windows(self):
        """Verify shortest path arrival windows across all 8,836 camera pairs."""
        reachable = 0
        for src in self.cam_ids:
            for tgt in self.cam_ids:
                if src == tgt:
                    self.assertEqual(self.engine.calculate_arrival_window(src, tgt), (0.0, 5.0))
                    continue
                t_min, t_max = self.engine.calculate_arrival_window(src, tgt)
                self.assertFalse(math.isinf(t_min), f"Unreachable camera pair: ({src}->{tgt})")
                self.assertGreater(t_min, 0.0)
                self.assertGreater(t_max, t_min)
                reachable += 1
        self.assertEqual(reachable, TOTAL_CAMERAS * (TOTAL_CAMERAS - 1))

    def test_sub_millisecond_query_latency_benchmark(self):
        """Benchmark 10,000 queries and confirm sub-millisecond execution."""
        n = 10000
        # 1. Arrival window queries (mix of direct and multi-hop)
        pairs = [(random.choice(self.cam_ids), random.choice(self.cam_ids)) for _ in range(n)]
        t0 = time.perf_counter()
        for u, v in pairs:
            _ = self.engine.calculate_arrival_window(u, v)
        t1 = time.perf_counter()
        mean_calc_us = ((t1 - t0) / n) * 1e6

        # 2. Transition validation queries
        trans = [(e["source_id"], e["target_id"], (e["t_min_s"] + e["t_max_s"]) / 2.0) for e in [random.choice(self.edges) for _ in range(n)]]
        t2 = time.perf_counter()
        for u, v, el in trans:
            _ = self.engine.is_valid_transition(u, v, el)
        t3 = time.perf_counter()
        mean_valid_us = ((t3 - t2) / n) * 1e6

        # Must be well under 1,000 microseconds (1 millisecond)
        self.assertLess(mean_calc_us, 1000.0, f"calculate_arrival_window too slow: {mean_calc_us:.2f} μs")
        self.assertLess(mean_valid_us, 1000.0, f"is_valid_transition too slow: {mean_valid_us:.2f} μs")


if __name__ == "__main__":
    unittest.main()
