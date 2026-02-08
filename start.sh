#!/usr/bin/env bash
# Start/restart all LibTrails services
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

API_PORT=8000
FRONTEND_PORT=4321

echo "=== LibTrails Service Manager ==="

# --- Stop existing services ---
echo ""
echo "Stopping existing services..."

api_pids=$(lsof -ti :$API_PORT 2>/dev/null || true)
if [ -n "$api_pids" ]; then
    echo "$api_pids" | xargs kill 2>/dev/null || true
    echo "  Stopped API server on port $API_PORT"
    sleep 1
else
    echo "  API server not running"
fi

# Astro may grab the next port if 4321 is busy, so check both
for port in $FRONTEND_PORT 4322 4323; do
    fe_pid=$(lsof -ti :$port 2>/dev/null || true)
    if [ -n "$fe_pid" ]; then
        kill $fe_pid 2>/dev/null || true
        echo "  Stopped frontend on port $port (pid $fe_pid)"
        sleep 1
    fi
done

# --- Start services ---
echo ""
echo "Starting services..."

# API server
uv run libtrails serve > /tmp/libtrails-api.log 2>&1 &
API_PID=$!
echo "  API server starting (pid $API_PID)..."

# Frontend
(cd "$PROJECT_DIR/web" && npm run dev > /tmp/libtrails-frontend.log 2>&1) &
FE_PID=$!
echo "  Frontend starting (pid $FE_PID)..."

# --- Wait for services to be ready ---
echo ""
echo "Waiting for services..."

for i in $(seq 1 15); do
    if curl -s -o /dev/null http://localhost:$API_PORT/api/v1/status 2>/dev/null; then
        echo "  API server ready on http://localhost:$API_PORT"
        break
    fi
    if [ $i -eq 15 ]; then
        echo "  API server may still be starting — check /tmp/libtrails-api.log"
    fi
    sleep 1
done

for i in $(seq 1 10); do
    fe_port=$(grep -oE 'localhost:[0-9]+' /tmp/libtrails-frontend.log 2>/dev/null | head -1 | sed 's/localhost://' || true)
    if [ -n "$fe_port" ]; then
        echo "  Frontend ready on http://localhost:$fe_port"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "  Frontend may still be starting — check /tmp/libtrails-frontend.log"
    fi
    sleep 1
done

echo ""
echo "=== All services started ==="
echo "  API docs: http://localhost:$API_PORT/docs"
echo "  Logs:     /tmp/libtrails-api.log"
echo "            /tmp/libtrails-frontend.log"
