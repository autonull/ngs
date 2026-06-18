#!/usr/bin/env bash
# NGS Dashboard Launch Script
# Usage: ./dashboard.sh [--simple|--demos] [--host 127.0.0.1] [--port 8050] [--debug]
#
# Dashboards:
#   (default)    Full Experiment Dashboard on port 8050
#   --simple     Streamlined dashboard on port 8051
#   --demos      Component Demos (3D visualizations) on port 8052
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
DEMOS_APP="${SCRIPT_DIR}/ngs/dashboard/demos_app.py"
HOST="0.0.0.0"
PORT="8050"
DEBUG=""
MODE="full"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --simple)
            MODE="simple"
            shift
            ;;
        --demos)
            MODE="demos"
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
            echo "Usage: $0 [--simple|--demos] [--host <host>] [--port <port>] [--debug]"
            echo ""
            echo "Modes:"
            echo "  (default)    Full Experiment Dashboard (port 8050)"
            echo "  --simple     Streamlined Experiment Dashboard (port 8051)"
            echo "  --demos      Component Demos - 3D Visualizations (port 8052)"
            echo ""
            echo "Options:"
            echo "  --host <host>   Host to bind to (default: 127.0.0.1)"
            echo "  --port <port>   Port to bind to (default varies by mode)"
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
case "$MODE" in
    simple)
        APP_PATH="${SIMPLE_APP}"
        [[ "$PORT" == "8050" ]] && PORT="8051"
        DASHBOARD_NAME="Simple Experiment Dashboard"
        MODULE="ngs.dashboard.simple_app"
        ;;
    demos)
        APP_PATH="${DEMOS_APP}"
        [[ "$PORT" == "8050" ]] && PORT="8052"
        DASHBOARD_NAME="Component Demos Dashboard"
        MODULE="ngs.dashboard.demos_app"
        ;;
    *)
        APP_PATH="${FULL_APP}"
        DASHBOARD_NAME="Full Experiment Dashboard"
        MODULE="ngs.dashboard.app"
        ;;
esac

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

# For demos, also need scikit-learn
if [[ "$MODE" == "demos" ]]; then
    if ! ${PYTHON} -c "import sklearn" 2>/dev/null; then
        echo "Installing scikit-learn for demos..."
        pip install scikit-learn
    fi
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
exec ${PYTHON} -m ${MODULE} --host="${HOST}" --port="${PORT}" ${DEBUG}
