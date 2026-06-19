#!/bin/bash
# Master script for continuous discovery experiments
# Usage: ./run_continuous_discovery.sh [--budget HOURS] [--concurrent N] [--quick]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUDGET=24
CONCURRENT=4
QUICK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --budget) BUDGET="$2"; shift 2 ;;
        --concurrent) CONCURRENT="$2"; shift 2 ;;
        --quick) QUICK=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ "$QUICK" = true ]; then
    BUDGET=0.1667  # 10 minutes
    CONCURRENT=1
    echo "Quick test mode: 10 minutes, 1 concurrent"
fi

echo "Starting Continuous Discovery..."
echo "  Budget: ${BUDGET}h"
echo "  Max concurrent: ${CONCURRENT}"
echo "  Variants: baseline, factorized, hyper, cfg_net"
echo "  Benchmarks: split_mnist, permuted_mnist, rotated_mnist, blurry_mnist, noisy_mnist"

python scripts/continuous_discovery.py \
    --budget "$BUDGET" \
    --concurrent "$CONCURRENT" \
    --variants baseline factorized hyper cfg_net \
    --benchmarks split_mnist permuted_mnist rotated_mnist blurry_mnist noisy_mnist \
    --seeds 42 123 456 \
    --epochs 5

# Generate final report
echo "Generating final report..."
python -c "
from scripts.results_db import ResultsDatabase
from scripts.insights import InsightStore, ReportGenerator, extract_insights

db = ResultsDatabase('./results_db')
store = InsightStore('./insights.jsonl')
insights = extract_insights(db)
for i in insights:
    store.add(i)
rg = ReportGenerator(db, store)
rg.generate_discovery_report('./discovery_report.md')
echo 'Report saved to discovery_report.md'
"
