#!/usr/bin/env bash
# scripts/start_daemon.sh
# Starts the BSW Kota Bogor MTMC Dashboard and ANE Background Indexing Daemon
# Uses macOS `caffeinate` to prevent system and network sleep during background tracking.

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

VENV_UVICORN="$DIR/.venv/bin/uvicorn"

if [ ! -f "$VENV_UVICORN" ]; then
    echo "Error: Virtualenv uvicorn not found at $VENV_UVICORN"
    exit 1
fi

# Stop existing server instance on port 8000 if running
PID=$(lsof -ti :8000 2>/dev/null || true)
if [ -n "$PID" ]; then
    echo "Stopping existing instance on port 8000 (PID: $PID)..."
    kill -TERM "$PID" 2>/dev/null || true
    sleep 1
fi

echo "Starting BSW Kota Bogor Traffic Intelligence Server with ANE Daemon..."
echo "Preventing macOS system sleep via caffeinate (-i -s)..."

# -i: Prevent system idle sleep
# -s: Prevent system sleep on AC power
nohup caffeinate -i -s "$VENV_UVICORN" backend.server:app --host 127.0.0.1 --port 8000 >> server.log 2>&1 &
SERVER_PID=$!

echo "Server started successfully in background with PID $SERVER_PID."
echo "Logs: $DIR/server.log"
echo "Dashboard: http://127.0.0.1:8000"
