#!/usr/bin/env zsh
# ==============================================================================
#  BSW Kota Bogor MTMC Traffic Intelligence Launcher
#  URL: http://127.0.0.1:8000
# ==============================================================================
set -e

# Prevent Errno 24 (Too many open files) with CoreML, OpenCV, and HLS sockets
ulimit -n 10240 2>/dev/null || ulimit -n 4096 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="$SCRIPT_DIR/.venv/bin/python3.11"

# Quiet OpenCV/FFmpeg decoder chatter (dead cameras are reported by the tracker instead)
export OPENCV_LOG_LEVEL="${OPENCV_LOG_LEVEL:-ERROR}"
export OPENCV_FFMPEG_LOGLEVEL="${OPENCV_FFMPEG_LOGLEVEL:-8}"
PORT=8000
# 0.0.0.0 serves the dashboard to the local network; HOST=127.0.0.1 ./start.sh keeps it local.
export HOST="${HOST:-0.0.0.0}"
export PORT
LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   BSW Kota Bogor MTMC Dashboard & Real-Time ANE Tracker          ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  URL         : http://127.0.0.1:$PORT                            ║"
echo "║  Python      : $PYTHON                                           ║"
echo "║  Hardware    : Apple Silicon Neural Engine (ANE)                 ║"
echo "║  AI Models   : YOLOv8m CoreML + ByteTrack (rolling, 96 cameras)  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Ensure the data directory exists
mkdir -p data

# Check if port 8000 is occupied and clear if needed
# Only the process LISTENING on the port: a plain `lsof -ti :$PORT` also lists every client
# connected to it (e.g. a browser tab), which made the kill fail -- or hit the browser instead.
PIDS=$(lsof -ti tcp:$PORT -sTCP:LISTEN 2>/dev/null || true)
if [ -n "$PIDS" ]; then
    echo "Stopping the server on port $PORT (PID: $PIDS)..."
    kill -TERM $PIDS 2>/dev/null || true
    # Graceful shutdown stops the tracker and its streams; allow up to 15 s, then force.
    for _ in $(seq 1 30); do
        lsof -ti tcp:$PORT -sTCP:LISTEN >/dev/null 2>&1 || break
        sleep 0.5
    done
    if lsof -ti tcp:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Still running; forcing it to stop."
        kill -9 $(lsof -ti tcp:$PORT -sTCP:LISTEN) 2>/dev/null || true
        sleep 0.5
    fi
fi

echo "🚀  Starting MTMC Traffic Dashboard Server..."
echo "    Open http://127.0.0.1:$PORT in your browser."
if [ "$HOST" = "0.0.0.0" ] && [ -n "$LAN_IP" ]; then
    echo "    On the local network: http://$LAN_IP:$PORT"
fi
echo "    Press Ctrl+C to stop."
echo ""

exec "$PYTHON" backend/server.py
