"""
Test-session isolation: every writable data file the backend opens at import time (traffic DB,
analytics DB, counting lines, free-flow speeds) is redirected to a temp dir before any test module
imports ``backend.server``. The suite used to open -- and migrate -- the live databases while the
dashboard server was running on them.
"""

import os
import shutil
import tempfile

# Extra city sources are opt-in per test (tests/test_camera_sources.py); the Bogor suite counts Bogor only.
os.environ["EXTRA_CAMERA_SOURCES"] = "0"

# Extra city sources are opt-in per test (tests/test_camera_sources.py); the Bogor suite counts Bogor only.
os.environ["EXTRA_CAMERA_SOURCES"] = "0"

_TMP = tempfile.mkdtemp(prefix="cctv-tests-")
for _var, _name in (("TRAFFIC_DB_PATH", "traffic_db.sqlite"),
                    ("ANALYTICS_DB_PATH", "traffic_timeseries.db"),
                    ("COUNTING_LINES_PATH", "counting_lines.json"),
                    ("FREE_FLOW_PATH", "free_flow_speeds.json"),
                    ("TRAFFIC_LANES_PATH", "traffic_lanes.npz")):
    os.environ[_var] = os.path.join(_TMP, _name)


def pytest_unconfigure(config):
    shutil.rmtree(_TMP, ignore_errors=True)
