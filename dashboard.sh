#!/usr/bin/env bash
# NGS Experiment Dashboard Launch Script
# Usage: ./dashboard.sh [--host 127.0.0.1] [--port 8050] [--debug]
#
# This script starts the NGS interactive experiment dashboard server
# at http://localhost:8050 (by default).
#
# Features:
# - Auto-opens browser (if available)
# - Validates Python environment
# - Installs missing dependencies if needed
# - Provides helpful error messages

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_PATH="${SCRIPT_DIR}/ngs/dashboard/app.py"
HOST="0.0.0.0"
PORT="8050"
DEBUG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
            echo "Usage: $0 [--host <host>] [--port <port>] [--debug]"
            echo ""
            echo "Options:"
            echo "  --host <host>   Host to bind to (default: 127.0.0.1)"
            echo "  --port <port>   Port to bind to (default: 8050)"
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
    shift
done

echo "======================================"
echo "  NGS Experiment Dashboard"
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
echo "Starting dashboard..."
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

# Start the Dash server
exec ${PYTHON} "${APP_PATH}" --host="${HOST}" --port="${PORT}" ${DEBUG}
