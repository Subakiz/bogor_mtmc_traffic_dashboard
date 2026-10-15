#!/usr/bin/env bash
# scripts/start_daemon.sh
# Starts the BSW Kota Bogor MTMC Dashboard and ANE Background Indexing Daemon
# Uses macOS `caffeinate` to prevent system and network sleep during background tracking.

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

VENV_UVICORN="$DIR/.venv/bin/uvicorn"
# 0.0.0.0 serves the dashboard to the local network; HOST=127.0.0.1 keeps it local.
HOST="${HOST:-0.0.0.0}"
LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)"
# Same environment as start.sh: enough file descriptors for 12 HLS streams, quiet FFmpeg/OpenCV.
ulimit -n 10240 2>/dev/null || ulimit -n 4096 2>/dev/null || true
export OPENCV_LOG_LEVEL="${OPENCV_LOG_LEVEL:-ERROR}"
export OPENCV_FFMPEG_LOGLEVEL="${OPENCV_FFMPEG_LOGLEVEL:-8}"

if [ ! -f "$VENV_UVICORN" ]; then
    echo "Error: Virtualenv uvicorn not found at $VENV_UVICORN"
    exit 1
fi

# Stop existing server instance on port 8000 if running
# Only the process LISTENING on the port: a plain `lsof -ti :8000` also lists every client
# connected to it (e.g. a browser tab), which made the kill fail -- or hit the browser instead.
PIDS=$(lsof -ti tcp:8000 -sTCP:LISTEN 2>/dev/null || true)
# Never signal an unrelated program that happens to hold the port.
for pid in $PIDS; do
    case "$(ps -o command= -p "$pid" 2>/dev/null)" in
        *backend/server.py*|*backend.server*) ;;
        *) echo "Port 8000 is held by another program (PID $pid: $(ps -o comm= -p "$pid" 2>/dev/null)), not this server. Not touching it; free the port first." >&2
           exit 1 ;;
    esac
done
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
fi

echo "Starting BSW Kota Bogor Traffic Intelligence Server with ANE Daemon..."
echo "Preventing macOS system sleep via caffeinate (-i -s)..."

# -i: Prevent system idle sleep
# -s: Prevent system sleep on AC power
nohup caffeinate -i -s "$VENV_UVICORN" backend.server:app --host "$HOST" --port 8000 >> server.log 2>&1 &
SERVER_PID=$!

echo "Server started successfully in background with PID $SERVER_PID."
echo "Logs: $DIR/server.log"
echo "Dashboard: http://127.0.0.1:8000"
if [ "$HOST" = "0.0.0.0" ] && [ -n "$LAN_IP" ]; then
    echo "On the local network: http://$LAN_IP:8000"
fi
