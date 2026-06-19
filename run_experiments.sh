#!/bin/bash
# Turn-key experiment runner for NGS paper results
# Usage: ./run_experiments.sh [suite] [--quick] [--dry-run]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default arguments
SUITE="${1:-all}"
QUICK=false
DRY_RUN=false
GENERATE_REPORT=false

# Parse arguments
shift 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick) QUICK=true ;;
        --dry-run) DRY_RUN=true ;;
        --generate-report) GENERATE_REPORT=true ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Build command
CMD="python scripts/reproduce.py $SUITE --output-dir ./paper_results --db-dir ./paper_db"

if [ "$QUICK" = true ]; then
    CMD="$CMD --quick"
fi

if [ "$DRY_RUN" = true ]; then
    CMD="$CMD --dry-run"
fi

if [ "$GENERATE_REPORT" = true ]; then
    CMD="$CMD --generate-report"
fi

echo "Running: $CMD"
eval $CMD

# If not dry run and not generate report, also generate final report
if [ "$DRY_RUN" = false ] && [ "$GENERATE_REPORT" = false ]; then
    echo ""
    echo "Generating paper report..."
    python scripts/reproduce.py --generate-report --output-dir ./paper_results --db-dir ./paper_db
fi
