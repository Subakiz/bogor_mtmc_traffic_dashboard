"""The shared bucketing rule: WIB-aligned buckets, intervals spread by overlap."""
import math
import unittest

from backend.bucketing import LOCAL_UTC_OFFSET_S, bucket_range, bucket_seconds, spread

OFF = LOCAL_UTC_OFFSET_S


class BucketingRuleTest(unittest.TestCase):
    def test_bucket_length_must_divide_a_day(self):
        self.assertEqual([bucket_seconds(m) for m in (1, 5, 15, 60, 1440)], [60, 300, 900, 3600, 86400])
        for bad in (0, 0.5, 7, 1441, "x", None):
            with self.assertRaises(ValueError):
                bucket_seconds(bad)

    def test_buckets_are_aligned_to_local_wall_clock(self):
        bs = 900
        start = 1_000_000.0
        k0, k1 = bucket_range(start, start + 4000, bs)
        first = k0 * bs - OFF
        self.assertGreaterEqual(first, start)
        self.assertEqual((first + OFF) % bs, 0)
        self.assertLessEqual(k1 * bs - OFF, start + 4000)       # only whole buckets inside the range

    def test_spread_shares_add_up_to_the_whole_interval(self):
        bs = 300
        s_ts, e_ts = 1_000_000.0 + 40, 1_000_000.0 + 700       # straddles several bucket edges
        k0, k1 = bucket_range(s_ts - 1000, e_ts + 1000, bs)
        parts = list(spread(s_ts, e_ts, k0, k1, bs))
        self.assertGreater(len(parts), 2)
        self.assertAlmostEqual(sum(f for _, f, _ in parts), 1.0)
        self.assertAlmostEqual(sum(o for _, _, o in parts), e_ts - s_ts)

    def test_spread_clips_to_the_requested_buckets(self):
        bs = 300
        s_ts, e_ts = 1_000_000.0, 1_000_000.0 + 900
        k0, k1 = bucket_range(s_ts, e_ts, bs)
        inner = list(spread(s_ts - 600, e_ts + 600, k0, k1, bs))
        self.assertEqual({k for k, _, _ in inner}, set(range(k0, k1)))

    def test_zero_length_interval_lands_in_its_own_bucket(self):
        bs = 300
        t = 1_000_000.0
        k = math.floor((t + OFF) / bs)
        self.assertEqual(list(spread(t, t, k - 1, k + 2, bs)), [(k, 1.0, 0.0)])
        self.assertEqual(list(spread(t, t, k + 1, k + 2, bs)), [])


if __name__ == "__main__":
    unittest.main()
