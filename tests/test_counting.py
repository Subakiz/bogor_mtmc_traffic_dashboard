"""
tests/test_counting.py
Counting lines learned from traffic motion, and when a single track is counted.
"""

import math
import os
import tempfile
import unittest

from backend import counting
from backend.counting import CountingLine, CountingLines, MotionModel, TrackCounter, ground_point


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


if __name__ == "__main__":
    unittest.main()
