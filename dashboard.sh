#!/usr/bin/env bash
# NGS Experiment Dashboard Launch Script
# Usage: ./dashboard.sh [--simple] [--host 127.0.0.1] [--port 8050] [--debug]
#
# This script starts the NGS experiment dashboard server.
# Default: Full dashboard on port 8050
# With --simple: Streamlined dashboard on port 8051
#
# Features:
# - Auto-opens browser (if available)
# - Validates Python environment
# - Installs missing dependencies if needed
# - Provides helpful error messages

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FULL_APP="${SCRIPT_DIR}/ngs/dashboard/app.py"
SIMPLE_APP="${SCRIPT_DIR}/ngs/dashboard/simple_app.py"
HOST="127.0.0.1"
PORT="8050"
DEBUG=""
SIMPLE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --simple)
            SIMPLE=true
            shift
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--simple] [--host <host>] [--port <port>] [--debug]"
            echo ""
            echo "Options:"
            echo "  --simple        Launch simple dashboard (default: full dashboard)"
            echo "  --host <host>   Host to bind to (default: 127.0.0.1)"
            echo "  --port <port>   Port to bind to (default: 8050 for full, 8051 for simple)"
            echo "  --debug         Enable Dash debug mode"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information."
            exit 1
            ;;
    esac
done

# Set defaults based on mode
if [[ "$SIMPLE" == true ]]; then
    APP_PATH="${SIMPLE_APP}"
    [[ "$PORT" == "8050" ]] && PORT="8051"
    DASHBOARD_NAME="Simple Experiment Dashboard"
else
    APP_PATH="${FULL_APP}"
    DASHBOARD_NAME="Full Experiment Dashboard"
fi

echo "======================================"
echo "  NGS ${DASHBOARD_NAME}"
echo "======================================"
echo ""

# Check Python availability
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH."
    exit 1
fi

PYTHON="$(python3 -c "import sys; print(sys.executable)" 2>/dev/null)"
echo "Python: ${PYTHON}"

# Check for required packages
echo "Checking dependencies..."
MISSING_DEPS=()

for pkg in dash plotly dash_bootstrap_components numpy; do
    if ! ${PYTHON} -c "import ${pkg}" 2>/dev/null; then
        MISSING_DEPS+=(${pkg})
    fi
done

if [[ ${#MISSING_DEPS[@]} -gt 0 ]]; then
    echo "Missing packages: ${MISSING_DEPS[*]}"
    echo "Installing..."
    pip install "${MISSING_DEPS[@]}"
fi

# Verify app.py exists
if [[ ! -f "${APP_PATH}" ]]; then
    echo "Error: Could not find dashboard app at ${APP_PATH}"
    exit 1
fi

echo ""
echo "Starting ${DASHBOARD_NAME}..."
echo "  URL:    http://${HOST}:${PORT}"
echo "  Debug:  ${DEBUG:-no}"
echo "  App:    ${APP_PATH}"
echo ""

# Open browser (macOS or Linux with xdg-open)
if command -v open &> /dev/null; then
    (sleep 2 && open "http://${HOST}:${PORT}") &
elif command -v xdg-open &> /dev/null; then
    (sleep 2 && xdg-open "http://${HOST}:${PORT}") &
fi

# Start the Dash server using module execution for proper imports
if [[ "$SIMPLE" == true ]]; then
    exec ${PYTHON} -m ngs.dashboard.simple_app --host="${HOST}" --port="${PORT}" ${DEBUG}
else
    exec ${PYTHON} -m ngs.dashboard.app --host="${HOST}" --port="${PORT}" ${DEBUG}
fi