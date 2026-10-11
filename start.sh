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
if lsof -ti :$PORT >/dev/null 2>&1; then
    echo "⚠  Port $PORT is already in use by PID: $(lsof -ti :$PORT)"
    echo "   Stopping existing instance..."
    kill -9 $(lsof -ti :$PORT) 2>/dev/null || true
    sleep 1
fi

echo "🚀  Starting MTMC Traffic Dashboard Server..."
echo "    Open http://127.0.0.1:$PORT in your browser."
echo "    Press Ctrl+C to stop."
echo ""

exec "$PYTHON" backend/server.py
