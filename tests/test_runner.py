#!/usr/bin/env python3
"""
tests/test_runner.py
BSW Kota Bogor Multi-Camera Tracking ("Follow That Car") Traffic Intelligence Dashboard.
Executable E2E Test Suite Runner for Tiers 1-4 (+ Tier 5 Adversarial).
"""

import argparse
import io
import json
import os
import sys
import time
import unittest
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is at top of sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TierResultTracker:
    def __init__(self, tier_name: str, tier_code: str):
        self.tier_name = tier_name
        self.tier_code = tier_code
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.errors = 0
        self.skipped = 0
        self.duration_s = 0.0
        self.failure_details: List[str] = []

    @property
    def is_success(self) -> bool:
        return self.failed == 0 and self.errors == 0


class StructuredTestRunner:
    """
    Discovers and executes tests across the 4-tier (+ Tier 5) test hierarchy.
    """

    TIER_DIRECTORIES = {
        "1": ("Tier 1 - Feature Coverage", "tests/tier1_features"),
        "2": ("Tier 2 - Boundary & Corner Cases", "tests/tier2_boundaries"),
        "3": ("Tier 3 - Cross-Feature Combinations", "tests/tier3_combinations"),
        "4": ("Tier 4 - Real-World City Scenarios", "tests/tier4_scenarios"),
        "5": ("Tier 5 - Adversarial Hardening", "tests/tier5_adversarial"),
    }

    def __init__(self, base_dir: Optional[str] = None, verbose: bool = False):
        self.base_dir = base_dir or PROJECT_ROOT
        if self.base_dir not in sys.path:
            sys.path.insert(0, self.base_dir)
        self.verbose = verbose

    def run_tier(self, tier_key: str) -> TierResultTracker:
        name, rel_path = self.TIER_DIRECTORIES[tier_key]
        abs_path = os.path.join(self.base_dir, rel_path)
        tracker = TierResultTracker(tier_name=name, tier_code=f"Tier {tier_key}")

        if not os.path.exists(abs_path):
            return tracker

        loader = unittest.TestLoader()
        suite = loader.discover(abs_path, pattern="test_*.py", top_level_dir=self.base_dir)
        tracker.total = suite.countTestCases()

        if tracker.total == 0:
            return tracker

        # Execute tests and capture results
        stream = io.StringIO() if not self.verbose else sys.stdout
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=2 if self.verbose else 1,
            resultclass=unittest.TextTestResult,
        )

        t0 = time.perf_counter()
        result = runner.run(suite)
        tracker.duration_s = time.perf_counter() - t0

        tracker.failed = len(result.failures)
        tracker.errors = len(result.errors)
        tracker.skipped = len(result.skipped)
        tracker.passed = tracker.total - tracker.failed - tracker.errors - tracker.skipped

        for failure in result.failures:
            tracker.failure_details.append(f"FAIL in {failure[0]}:\n{failure[1]}")
        for error in result.errors:
            tracker.failure_details.append(f"ERROR in {error[0]}:\n{error[1]}")

        return tracker

    def run_all(self, selected_tier: Optional[str] = None) -> Tuple[List[TierResultTracker], bool]:
        tiers_to_run = []
        if selected_tier and selected_tier.lower() != "all":
            clean_key = selected_tier.replace("tier", "").replace("Tier", "").strip()
            if clean_key in self.TIER_DIRECTORIES:
                tiers_to_run = [clean_key]
            else:
                print(f"Error: Unknown tier '{selected_tier}'. Valid tiers: 1, 2, 3, 4, 5, all", file=sys.stderr)
                sys.exit(1)
        else:
            # Default to all available tiers (1 to 4, and 5 if present)
            for k, (_, rel_path) in self.TIER_DIRECTORIES.items():
                if os.path.exists(os.path.join(self.base_dir, rel_path)):
                    tiers_to_run.append(k)

        results = []
        overall_success = True
        for t_key in tiers_to_run:
            res = self.run_tier(t_key)
            results.append(res)
            if not res.is_success:
                overall_success = False

        return results, overall_success

    def print_ascii_summary(self, results: List[TierResultTracker]):
        total_tests = sum(r.total for r in results)
        total_passed = sum(r.passed for r in results)
        total_failed = sum(r.failed for r in results)
        total_errors = sum(r.errors for r in results)
        total_duration = sum(r.duration_s for r in results)

        print("\n" + "=" * 90)
        print(" BSW KOTA BOGOR MTMC INTELLIGENCE DASHBOARD - E2E TEST SUITE REPORT")
        print("=" * 90)
        print(f" {'TIER':<10} | {'NAME':<36} | {'TESTS':<6} | {'PASS':<6} | {'FAIL':<6} | {'TIME (s)':<8} | {'STATUS'}")
        print("-" * 90)

        for r in results:
            status_str = "PASSED" if r.is_success else "FAILED"
            print(
                f" {r.tier_code:<10} | {r.tier_name:<36} | {r.total:<6} | {r.passed:<6} | {r.failed + r.errors:<6} | {r.duration_s:<8.3f} | {status_str}"
            )

        print("-" * 90)
        overall_status = "ALL TESTS PASSED (100%)" if (total_failed == 0 and total_errors == 0) else "TEST FAILURES DETECTED"
        print(
            f" {'TOTAL':<10} | {'All Executed Test Suites':<36} | {total_tests:<6} | {total_passed:<6} | {total_failed + total_errors:<6} | {total_duration:<8.3f} | {overall_status}"
        )
        print("=" * 90)

        # Print failure details if any
        failures_exist = False
        for r in results:
            if r.failure_details:
                failures_exist = True
                print(f"\n--- Failure Details in {r.tier_code}: {r.tier_name} ---")
                for detail in r.failure_details:
                    print(detail)

        if not failures_exist:
            print(f"\n>> SUCCESS: 100% of {total_tests} test cases passed across {len(results)} tiers in {total_duration:.3f}s.\n")

    def export_json_summary(self, results: List[TierResultTracker]) -> str:
        report = {
            "project": "BSW Kota Bogor MTMC Traffic Intelligence Dashboard",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_tests": sum(r.total for r in results),
            "passed": sum(r.passed for r in results),
            "failed": sum(r.failed + r.errors for r in results),
            "total_duration_s": round(sum(r.duration_s for r in results), 4),
            "pass_rate_percent": round(
                (sum(r.passed for r in results) / max(1, sum(r.total for r in results))) * 100.0, 2
            ),
            "tiers": [
                {
                    "tier_code": r.tier_code,
                    "tier_name": r.tier_name,
                    "total": r.total,
                    "passed": r.passed,
                    "failed": r.failed,
                    "errors": r.errors,
                    "duration_s": round(r.duration_s, 4),
                    "status": "PASSED" if r.is_success else "FAILED",
                }
                for r in results
            ],
        }
        return json.dumps(report, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Run BSW Kota Bogor Multi-Camera Tracking E2E Test Suite."
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="all",
        help="Specific tier to run: 1, 2, 3, 4, 5, or all (default: all)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose test execution output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON summary",
    )

    args = parser.parse_args()

    runner = StructuredTestRunner(verbose=args.verbose)
    results, success = runner.run_all(selected_tier=args.tier)

    if args.json:
        print(runner.export_json_summary(results))
    else:
        runner.print_ascii_summary(results)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
