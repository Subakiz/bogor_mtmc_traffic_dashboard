"""
tests/test_counting.py
Counting lines learned from traffic motion, and when a single track is counted.
"""

import json
import math
import os
import tempfile
import unittest

from backend import counting
from backend.config import DATA_DIR
from backend.counting import CountingLine, CountingLines, MotionModel, TrackCounter, ground_point, load_overrides
from tests import _topology


def teach(model, trajectories):
    for start, end in trajectories:
        model.add(start, end)


VERTICAL = [((0.5, 0.2), (0.5, 0.8))] * 6 + [((0.45, 0.85), (0.45, 0.25))] * 6     # two opposite lanes
HORIZONTAL = [((0.1, 0.6), (0.9, 0.6))] * 12


class MotionModelTest(unittest.TestCase):
    def test_learns_vertical_axis_from_opposite_lanes(self):
        m = MotionModel()
        teach(m, VERTICAL)
        self.assertIsNotNone(m.line)
        self.assertAlmostEqual(abs(m.line.ny), 1.0, places=3)   # normal points along the road
        self.assertAlmostEqual(m.line.py, 0.525, places=2)      # through the middle of the traffic
        self.assertEqual(m.line.labels(), ("towards camera", "away from camera"))

    def test_learns_horizontal_axis(self):
        m = MotionModel()
        teach(m, HORIZONTAL)
        self.assertAlmostEqual(m.line.nx, 1.0, places=3)
        self.assertAlmostEqual(m.line.px, 0.5, places=3)
        self.assertEqual(m.line.labels(), ("left to right", "right to left"))

    def test_line_follows_a_re_aimed_camera(self):
        m = MotionModel()
        teach(m, HORIZONTAL * 100)                                  # a long history in one view
        teach(m, VERTICAL * 125)                                    # then the camera is turned
        self.assertGreater(abs(m.line.ny), abs(m.line.nx))
        self.assertEqual(m.tracks, counting.LINE_MEMORY)

    def test_needs_enough_trajectories_and_ignores_short_ones(self):
        m = MotionModel()
        teach(m, HORIZONTAL[: counting.LINE_MIN_TRACKS - 1])
        self.assertIsNone(m.line)
        self.assertFalse(m.add((0.5, 0.5), (0.52, 0.5)))        # parked / jitter
        self.assertIsNone(m.line)
        m.add((0.1, 0.6), (0.9, 0.6))
        self.assertIsNotNone(m.line)

    def test_line_endpoints_span_the_frame(self):
        m = MotionModel()
        teach(m, HORIZONTAL)
        (x1, y1), (x2, y2) = m.line.endpoints()
        for v in (x1, y1, x2, y2):
            self.assertGreaterEqual(v, -1e-6)
            self.assertLessEqual(v, 1 + 1e-6)
        self.assertAlmostEqual(math.hypot(x2 - x1, y2 - y1), 1.0, places=3)   # vertical line, full height


class TrackCounterTest(unittest.TestCase):
    def setUp(self):
        self.model = MotionModel()
        teach(self.model, VERTICAL)          # line across the road at y ~ 0.525, normal along +y

    def walk(self, counter, ys, x=0.5, can_count=True):
        return [counter.update((x, y), self.model, can_count) for y in ys]

    def test_counts_once_with_direction(self):
        c = TrackCounter((0.5, 0.2))
        hits = self.walk(c, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        self.assertEqual(hits.count(True), 1)
        self.assertEqual((c.direction, c.mode), (1, "line"))

        back = TrackCounter((0.5, 0.8))
        self.assertEqual(self.walk(back, [0.8, 0.6, 0.4, 0.2]).count(True), 1)
        self.assertEqual(back.direction, -1)

    def test_jitter_at_the_line_and_crossing_back_do_not_recount(self):
        c = TrackCounter((0.5, 0.4))
        hits = self.walk(c, [0.4, 0.52, 0.53, 0.52, 0.6, 0.45, 0.6, 0.45])
        self.assertEqual(hits.count(True), 1)

    def test_vehicle_first_seen_past_the_line_is_not_counted(self):
        c = TrackCounter((0.5, 0.7))
        self.assertNotIn(True, self.walk(c, [0.7, 0.8, 0.9]))

    def test_crossing_before_confirmation_counts_once_confirmed(self):
        c = TrackCounter((0.5, 0.3))
        self.assertNotIn(True, self.walk(c, [0.3, 0.6], can_count=False))
        self.assertEqual(self.walk(c, [0.65]), [True])

    def test_passage_mode_before_a_line_is_learned(self):
        fresh = MotionModel()
        moving, parked = TrackCounter((0.1, 0.5)), TrackCounter((0.5, 0.5))
        moved = [moving.update((0.1 + 0.05 * i, 0.5), fresh, True) for i in range(1, 6)]
        self.assertEqual(moved.count(True), 1)
        self.assertEqual(moving.mode, "passage")
        self.assertNotIn(True, [parked.update((0.5 + 0.001 * i, 0.5), fresh, True) for i in range(50)])


def refit_until(model, start, end, px_at_least):
    """Feeds trajectories until the fitted line's position passes ``px_at_least``."""
    for _ in range(200):
        if model.line.px >= px_at_least:
            return
        model.add(start, end)
    raise AssertionError("line never moved")


class RefitMidTrackTest(unittest.TestCase):
    """Departures refit the counting line while other tracks of the same visit are still alive."""

    def setUp(self):
        self.model = MotionModel()
        teach(self.model, [((0.1, 0.6), (0.7, 0.6))] * counting.LINE_MIN_TRACKS)   # line at x = 0.40
        self.old_px = self.model.line.px
        self.assertAlmostEqual(self.old_px, 0.40, places=2)

    def test_queued_car_is_not_counted_when_the_line_moves_across_it(self):
        # Regression: first_side was judged against the line as it was when the track appeared, so a
        # standing car at x = 0.43 was "on the other side" once the line moved to ~0.47.
        car = TrackCounter((0.43, 0.6))
        self.assertFalse(car.update((0.43, 0.6), self.model, True))
        self.assertGreater(self.model.line.side((0.43, 0.6)), 0, "ahead of the original line")
        refit_until(self.model, (0.4, 0.6), (0.9, 0.6), 0.465)
        self.assertLess(self.model.line.side((0.43, 0.6)), 0, "behind the refitted line")
        hits = [car.update((0.43, 0.6), self.model, True) for _ in range(5)]
        self.assertNotIn(True, hits)
        self.assertFalse(car.counted)

    def test_jitter_of_a_standing_car_is_not_counted_after_a_refit(self):
        car = TrackCounter((0.43, 0.6))
        car.update((0.43, 0.6), self.model, True)
        refit_until(self.model, (0.4, 0.6), (0.9, 0.6), 0.465)
        hits = [car.update((0.43 + d, 0.6), self.model, True) for d in (0.004, -0.004, 0.006, -0.005)]
        self.assertNotIn(True, hits)

    def test_real_crossing_is_still_counted_once_after_a_refit(self):
        car = TrackCounter((0.2, 0.6))
        self.assertFalse(car.update((0.2, 0.6), self.model, True))
        self.assertFalse(car.update((0.3, 0.6), self.model, True))
        refit_until(self.model, (0.4, 0.6), (0.9, 0.6), 0.465)
        hits = [car.update((x, 0.6), self.model, True) for x in (0.4, 0.5, 0.6, 0.7, 0.4, 0.7)]
        self.assertEqual(hits.count(True), 1)
        self.assertEqual((car.direction, car.mode), (1, "line"))

    def test_a_car_must_have_moved_to_be_counted(self):
        # Start and current position must be on opposite sides of the *current* line, which is
        # impossible without moving at least twice the hysteresis band.
        car = TrackCounter((0.405, 0.6))
        self.assertFalse(car.update((0.405, 0.6), self.model, True))
        self.assertFalse(car.update((0.412, 0.6), self.model, True))
        self.assertFalse(car.counted)

    def test_track_starting_on_the_line_adopts_its_first_clear_side(self):
        car = TrackCounter((0.40, 0.6))                       # on the line: no side yet
        hits = [car.update((x, 0.6), self.model, True) for x in (0.40, 0.45, 0.6, 0.5)]
        self.assertEqual(hits, [False] * 4, "drove away from the line first, never crossed it")
        back = TrackCounter((0.40, 0.6))
        hits = [back.update((x, 0.6), self.model, True) for x in (0.40, 0.30, 0.50)]
        self.assertEqual(hits.count(True), 1)
        self.assertEqual(back.direction, 1)


class CountingLinesPersistenceTest(unittest.TestCase):
    def test_learned_lines_survive_a_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lines.json")
            lines = CountingLines(path)
            for start, end in HORIZONTAL:
                lines.learn(167, start, end)
            lines.save()
            restored = CountingLines(path).line(167)
            self.assertIsNotNone(restored)
            self.assertAlmostEqual(restored.nx, lines.line(167).nx)
            self.assertIsNone(CountingLines(path).line(79))

    def test_ground_point_is_bottom_centre(self):
        self.assertEqual(ground_point((100, 50, 300, 250), 400, 500), (0.5, 0.5))


WIDE = 16 / 9


def pixel_trajectory(deg, aspect, centre=(0.5, 0.5), length=0.3):
    """A trajectory at ``deg`` degrees in the image (pixels), as 0..1 coordinates of a frame
    ``aspect`` wide per unit of height."""
    t = math.radians(deg)
    dx, dy = math.cos(t) * length / aspect, math.sin(t) * length
    return (centre[0] - dx / 2, centre[1] - dy / 2), (centre[0] + dx / 2, centre[1] + dy / 2)


def pixel_angle_between_line_and(line, deg, aspect):
    """Angle in pixels between the drawn line and a travel direction of ``deg`` degrees."""
    (x1, y1), (x2, y2) = line.endpoints()
    lx, ly = (x2 - x1) * aspect, y2 - y1
    t = math.radians(deg)
    cos = abs(lx * math.cos(t) + ly * math.sin(t)) / math.hypot(lx, ly)
    return math.degrees(math.acos(min(1.0, cos)))


class PixelSpaceAxisTest(unittest.TestCase):
    def test_line_is_perpendicular_to_traffic_in_pixels(self):
        m = MotionModel()
        m.set_aspect(WIDE)
        teach(m, [pixel_trajectory(30, WIDE)] * 12)
        self.assertAlmostEqual(pixel_angle_between_line_and(m.line, 30, WIDE), 90.0, places=3)

    def test_legacy_model_keeps_its_line(self):
        legacy = MotionModel()
        teach(legacy, [pixel_trajectory(30, WIDE)] * 12)
        stored = {k: v for k, v in legacy.to_dict().items() if k not in ("aspect", "ref")}
        m = MotionModel.from_dict(stored)
        before = (m.line.px, m.line.py, m.line.nx, m.line.ny)
        self.assertFalse(m.set_aspect(WIDE))
        m.add(*pixel_trajectory(30, WIDE))
        self.assertIsNone(m.aspect)
        self.assertAlmostEqual(m.line.nx, before[2])
        self.assertAlmostEqual(m.line.ny, before[3])

    def test_resolution_change_keeps_direction_coherence_and_sign(self):
        # Learned on a square stream, which then switches to 16:9.
        legacy = MotionModel()
        legacy.set_aspect(1.0)
        teach(legacy, [pixel_trajectory(30, WIDE)] * 12 + [pixel_trajectory(80, WIDE)] * 4)
        stored = legacy.to_dict()
        old_normal = (legacy.line.nx, legacy.line.ny)
        old_travel = legacy.axis()                     # unit travel direction in 0..1 coordinates
        m = MotionModel.from_dict(stored)
        coherence = m.coherence
        self.assertTrue(m.set_aspect(WIDE))
        self.assertFalse(m.set_aspect(WIDE))
        self.assertAlmostEqual(m.coherence, coherence, places=9)
        # The axis still points the same way in the image, and the line is now perpendicular to it.
        travel_deg = math.degrees(math.atan2(old_travel[1], old_travel[0] * WIDE))
        self.assertAlmostEqual(pixel_angle_between_line_and(m.line, travel_deg, WIDE), 90.0, places=3)
        self.assertGreater(m.line.nx * old_normal[0] + m.line.ny * old_normal[1], 0, "same + direction")
        restored = MotionModel.from_dict(json.loads(json.dumps(m.to_dict())))
        self.assertEqual(restored.aspect, WIDE)
        self.assertAlmostEqual(restored.line.nx, m.line.nx)

    def test_direction_sign_is_stable_across_45_degrees(self):
        # The old sign rule (+y if mostly vertical, else +x) flipped + and - when the axis crossed 45 deg.
        m = MotionModel()
        teach(m, [pixel_trajectory(-40, 1.0)] * 12)
        first = (m.line.nx, m.line.ny)
        for deg in range(-41, -56, -1):
            teach(m, [pixel_trajectory(deg, 1.0)] * 40)
            self.assertGreater(m.line.nx * first[0] + m.line.ny * first[1], 0, f"sign flipped at {deg} deg")
        self.assertGreater(abs(m.line.ny), abs(m.line.nx), "the axis did cross 45 deg")
        restored = MotionModel.from_dict(json.loads(json.dumps(m.to_dict())))
        self.assertEqual((restored.line.nx, restored.line.ny), (m.line.nx, m.line.ny))

    def test_labels_follow_the_sign(self):
        self.assertEqual(CountingLine(0.5, 0.5, 0.0, -1.0).labels(), ("away from camera", "towards camera"))
        self.assertEqual(CountingLine(0.5, 0.5, -1.0, 0.0).labels(), ("right to left", "left to right"))


class CoherenceGateTest(unittest.TestCase):
    def test_incoherent_traffic_falls_back_to_passage_mode(self):
        m = MotionModel()
        teach(m, HORIZONTAL + [((0.5, 0.1), (0.5, 0.9))] * 11)      # a junction: flows (almost) cancel
        self.assertLess(m.coherence, counting.COHERENCE_MIN)
        self.assertIsNotNone(m.learned)
        self.assertIsNone(m.line)
        self.assertEqual(m.quality(), "passage")
        c = TrackCounter((0.1, 0.3))
        hits = [c.update((0.1 + 0.05 * i, 0.3), m, True) for i in range(1, 6)]
        self.assertEqual(hits.count(True), 1)
        self.assertEqual(c.mode, "passage")

    def test_weakly_coherent_line_is_kept_but_flagged_low(self):
        m = MotionModel()
        teach(m, HORIZONTAL * 3 + [((0.5, 0.1), (0.5, 0.9))] * 18)
        self.assertTrue(counting.COHERENCE_MIN <= m.coherence < counting.COHERENCE_LOW, m.coherence)
        self.assertIsNotNone(m.line)
        self.assertEqual(m.quality(), "low")
        clean = MotionModel()
        teach(clean, HORIZONTAL)
        self.assertEqual(clean.quality(), "learned")


class ManualLineTest(unittest.TestCase):
    def setUp(self):
        self.model = MotionModel()
        self.model.manual = CountingLine.through((0.25, 0.6), (0.75, 0.6), segment=True)

    def walk(self, start, points):
        c = TrackCounter(start)
        return c, [c.update(p, self.model, True) for p in [start] + points]

    def test_left_to_right_line_counts_towards_camera_as_positive(self):
        c, hits = self.walk((0.5, 0.4), [(0.5, 0.5), (0.5, 0.7)])
        self.assertEqual(hits.count(True), 1)
        self.assertEqual((c.direction, c.mode), (1, "line"))
        self.assertEqual(self.model.line.labels(), ("towards camera", "away from camera"))
        self.assertEqual(self.model.quality(), "manual")

    def test_segment_counts_only_crossings_between_its_ends(self):
        _, hits = self.walk((0.1, 0.4), [(0.1, 0.7)])                  # beyond the left end
        self.assertNotIn(True, hits)
        # Crossed beyond the end, then drove along under the segment: still not a crossing.
        _, hits = self.walk((0.9, 0.4), [(0.9, 0.7), (0.6, 0.7), (0.5, 0.75)])
        self.assertNotIn(True, hits)
        _, hits = self.walk((0.74, 0.4), [(0.74, 0.7)])                # just inside the right end
        self.assertEqual(hits.count(True), 1)

    def test_crossing_beyond_the_end_before_confirmation_is_not_counted_later(self):
        c = TrackCounter((0.9, 0.4))
        for p in [(0.9, 0.4), (0.9, 0.7), (0.6, 0.7)]:               # not yet confirmed
            self.assertFalse(c.update(p, self.model, False))
        self.assertFalse(c.update((0.5, 0.75), self.model, True))
        self.assertFalse(c.counted)

    def test_manual_line_wins_over_a_learned_one(self):
        teach(self.model, VERTICAL)
        self.assertIs(self.model.line, self.model.manual)
        self.assertEqual(self.model.line.endpoints(), ((0.75, 0.6), (0.25, 0.6)))

    def test_excluded_camera_never_counts(self):
        self.model.excluded = True
        teach(self.model, VERTICAL)
        _, hits = self.walk((0.5, 0.2), [(0.5, 0.5), (0.5, 0.8)])
        self.assertNotIn(True, hits)
        self.model.manual = None
        _, hits = self.walk((0.1, 0.2), [(0.5, 0.5), (0.9, 0.9)])   # no passage counting either
        self.assertNotIn(True, hits)
        self.assertEqual(self.model.quality(), "excluded")


class OverrideFileTest(unittest.TestCase):
    def write(self, tmp, data):
        path = os.path.join(tmp, "overrides.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return path

    def test_overrides_apply_without_a_learned_model_and_are_not_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            overrides = self.write(tmp, {"_doc": "x", "version": 1,
                                         "193": {"line": [0.25, 0.62, 1.0, 0.62], "segment": True},
                                         "159": {"exclude": True},
                                         "86": {"line": [0.2, 0.6, 0.2, 0.6]},            # degenerate
                                         "87": {"line": [0.2, 0.6, 1.4, 0.6]},            # off frame
                                         "88": {"line": [0, 0.5, 1, 0.5], "exclude": True},
                                         "abc": {"exclude": True}})
            path = os.path.join(tmp, "lines.json")
            lines = CountingLines(path, overrides)
            self.assertEqual(lines.line(193).half, 0.375)
            self.assertEqual(lines.quality(193), "manual")
            self.assertIsNone(lines.line(159))
            self.assertTrue(lines.excluded(159))
            self.assertEqual(lines.quality(159), "excluded")
            for bad in (86, 87, 88):
                self.assertIsNone(lines.line(bad))
                self.assertFalse(lines.excluded(bad))
            self.assertEqual(lines.quality(79), "passage")
            for start, end in HORIZONTAL:
                lines.learn(193, start, end)
            lines.save()
            with open(path, encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(set(saved), {"193"})
            self.assertNotIn("manual", saved["193"])
            self.assertIsNotNone(CountingLines(path).line(193), "learned line kept, without override")
            self.assertEqual(CountingLines(path, overrides).quality(193), "manual")

    def test_unreadable_file_applies_no_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "overrides.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("{not json")
            self.assertEqual(load_overrides(path), {})
            self.assertEqual(load_overrides(os.path.join(tmp, "missing.json")), {})

    def test_shipped_override_file_is_valid(self):
        path = os.path.join(DATA_DIR, "counting_line_overrides.json")
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        ids = {int(k) for k in raw if not k.startswith("_") and k != "version"}
        loaded = load_overrides(path)
        self.assertEqual(set(loaded), ids, "every entry must load")
        self.assertLessEqual(ids, {int(n["id"]) for n in _topology["nodes"]}, "override for an unknown camera")
        self.assertIn(193, ids)
        self.assertTrue(loaded[159]["exclude"])


if __name__ == "__main__":
    unittest.main()
