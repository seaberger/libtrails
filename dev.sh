#!/bin/bash
# Start both API and frontend dev servers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Kill any existing processes on our ports
cleanup_ports() {
    echo -e "${YELLOW}Checking for processes on ports 8000 and 4321...${NC}"

    # Kill processes on port 8000 (API)
    lsof -ti :8000 2>/dev/null | xargs kill -9 2>/dev/null || true

    # Kill processes on port 4321 (Frontend)
    lsof -ti :4321 2>/dev/null | xargs kill -9 2>/dev/null || true

    sleep 1
}

# Cleanup function for script exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down servers...${NC}"
    kill $API_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

# Set up trap for clean exit
trap cleanup SIGINT SIGTERM

# Clean up ports first
cleanup_ports

echo -e "${GREEN}Starting LibTrails dev servers...${NC}\n"

# Start API server in background
echo -e "${GREEN}[API]${NC} Starting on http://localhost:8000"
uv run libtrails serve --reload &
API_PID=$!

# Wait a moment for API to start
sleep 2

# Start frontend dev server in background
echo -e "${GREEN}[Frontend]${NC} Starting on http://localhost:4321"
cd web && npm run dev &
FRONTEND_PID=$!

echo -e "\n${GREEN}Both servers running!${NC}"
echo -e "  API:      http://localhost:8000"
echo -e "  Frontend: http://localhost:4321"
echo -e "\nPress Ctrl+C to stop both servers.\n"

# Wait for both processes
wait
