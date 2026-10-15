#!/usr/bin/env bash
# scripts/stop_daemon.sh
# Gracefully stops the BSW Kota Bogor Dashboard and ANE Background Indexing Daemon

set -e

# Only the process LISTENING on the port: a plain `lsof -ti :8000` also lists every client
# connected to it (e.g. a browser tab), which made the kill fail -- or hit the browser instead.
PIDS=""
for pid in $(lsof -ti tcp:8000 -sTCP:LISTEN 2>/dev/null || true); do
    case "$(ps -o command= -p "$pid" 2>/dev/null)" in
        *backend/server.py*|*backend.server*) PIDS="$PIDS $pid" ;;
        *) echo "Skipping PID $pid on port 8000 ($(ps -o comm= -p "$pid" 2>/dev/null)): not this server." ;;
    esac
done
alive() { for p in $PIDS; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
if [ -n "$PIDS" ]; then
    echo "Stopping the server on port 8000 (PID:$PIDS)..."
    kill -TERM $PIDS 2>/dev/null || true
    # Graceful shutdown stops the tracker and its streams; allow up to 15 s, then force.
    for _ in $(seq 1 30); do
        alive || break
        sleep 0.5
    done
    if alive; then
        echo "Still running; forcing it to stop."
        kill -9 $PIDS 2>/dev/null || true
        sleep 0.5
    fi
    echo "Stopped."
else
    echo "No server running on port 8000."
fi
