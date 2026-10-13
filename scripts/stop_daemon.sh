#!/usr/bin/env bash
# scripts/stop_daemon.sh
# Gracefully stops the BSW Kota Bogor Dashboard and ANE Background Indexing Daemon

set -e

# Only the process LISTENING on the port: a plain `lsof -ti :8000` also lists every client
# connected to it (e.g. a browser tab), which made the kill fail -- or hit the browser instead.
PIDS=$(lsof -ti tcp:8000 -sTCP:LISTEN 2>/dev/null || true)
if [ -n "$PIDS" ]; then
    echo "Stopping the server on port 8000 (PID: $PIDS)..."
    kill -TERM $PIDS 2>/dev/null || true
    # Graceful shutdown stops the tracker and its streams; allow up to 15 s, then force.
    for _ in $(seq 1 30); do
        lsof -ti tcp:8000 -sTCP:LISTEN >/dev/null 2>&1 || break
        sleep 0.5
    done
    if lsof -ti tcp:8000 -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Still running; forcing it to stop."
        kill -9 $(lsof -ti tcp:8000 -sTCP:LISTEN) 2>/dev/null || true
        sleep 0.5
    fi
    echo "Stopped."
else
    echo "No server running on port 8000."
fi
