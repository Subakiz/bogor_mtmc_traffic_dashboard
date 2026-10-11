#!/usr/bin/env bash
# scripts/stop_daemon.sh
# Gracefully stops the BSW Kota Bogor Dashboard and ANE Background Indexing Daemon

set -e

PID=$(lsof -ti :8000 2>/dev/null || true)
if [ -n "$PID" ]; then
    echo "Stopping server on port 8000 (PID: $PID)..."
    kill -TERM "$PID" 2>/dev/null || true
    sleep 1
    # Check if still running
    if kill -0 "$PID" 2>/dev/null; then
        echo "Force killing remaining process..."
        kill -9 "$PID" 2>/dev/null || true
    fi
    echo "Stopped."
else
    echo "No server running on port 8000."
fi
