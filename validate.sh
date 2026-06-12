#!/bin/bash
# MNGS Validation Script - Run the full experimental validation matrix
# Usage: ./validate.sh [stage]
# Stages: smoke, m3_2, m3_3, m3_3_lora, m3_4, m3_5, m3_6, m3_7, all, param_matched

set -euo pipefail

STAGE="${1:-all}"
OUT_DIR="./results"
PLOTS_DIR="./plots"
SEEDS=(42 123 456)

mkdir -p "$OUT_DIR" "$PLOTS_DIR"

# Experiment list (11 datasets)
EXPERIMENTS=(
    split_mnist
    split_fashion
    permuted_mnist
    rotated_mnist
    blurry_mnist
    noisy_mnist
    split_cifar10
    split_cifar100
    digits
    split_cifar100_20
    full_mnist
)

# Model groups - parameter-matched (for fair comparison with baselines ~513K params)
MODELS_MNGS_PM=("mngs_baseline" "mngs_cfg_net" "mngs_abl_hyper")
# Model groups - LoRA efficient versions
MODELS_MNGS_LORA=("mngs_baseline_lora" "mngs_cfg_net_lora" "mngs_abl_hyper_lora")
# Baselines
MODELS_LEAN=("lean_ngs")
MODELS_BASELINES=("mlp" "er" "ewc" "si" "lwf" "lora")

EXP_STR="${EXPERIMENTS[*]}"

case "$STAGE" in
    smoke)
        echo "=== SMOKE TEST: 1 epoch, 1 seed, 1 dataset ==="
        python -m experiments.main \
            --experiments split_mnist \
            --models lean_ngs \
            --seeds 42 \
            --output-dir "$OUT_DIR" \
            --plots-dir "$PLOTS_DIR" \
            --no-verbose
        ;;

    param_matched)
        echo "=== PARAMETER-MATCHED COMPARISON: MNGS (full adapters) vs Baselines ==="
        echo "MNGS profiles use max_k=448, use_lora=False for ~513K params matching MLP [512,256]"
        python -m experiments.main \
            --experiments $EXP_STR \
            --models ${MODELS_MNGS_PM[*]} ${MODELS_BASELINES[*]} \
            --seeds ${SEEDS[*]} \
            --output-dir "$OUT_DIR" \
            --plots-dir "$PLOTS_DIR"
        ;;

    m3_2)
        echo "=== M3.2: Baseline LeanNGS on all datasets ==="
        python -m experiments.main \
            --experiments $EXP_STR \
            --models ${MODELS_LEAN[*]} \
            --seeds ${SEEDS[*]} \
            --output-dir "$OUT_DIR" \
            --plots-dir "$PLOTS_DIR"
        ;;

    m3_3)
        echo "=== M3.3: All MNGS profiles (param-matched) on all datasets ==="
        python -m experiments.main \
            --experiments $EXP_STR \
            --models ${MODELS_MNGS_PM[*]} \
            --seeds ${SEEDS[*]} \
            --output-dir "$OUT_DIR" \
            --plots-dir "$PLOTS_DIR"
        ;;

    m3_3_lora)
        echo "=== M3.3 LoRA: All MNGS profiles (LoRA efficient) on all datasets ==="
        python -m experiments.main \
            --experiments $EXP_STR \
            --models ${MODELS_MNGS_LORA[*]} \
            --seeds ${SEEDS[*]} \
            --output-dir "$OUT_DIR" \
            --plots-dir "$PLOTS_DIR"
        ;;

    m3_4)
        echo "=== M3.4: All baselines on all datasets ==="
        python -m experiments.main \
            --experiments $EXP_STR \
            --models ${MODELS_BASELINES[*]} \
            --seeds ${SEEDS[*]} \
            --output-dir "$OUT_DIR" \
            --plots-dir "$PLOTS_DIR"
        ;;

    m3_5)
        echo "=== M3.5: Structural ablation (requires M3.2 + M3.3 results) ==="
        python -m experiments.ablation \
            --results-dir "$OUT_DIR" \
            --output-dir "$OUT_DIR/ablation"
        ;;

    m3_6)
        echo "=== M3.6: HPO per dataset (50 trials each) ==="
        for exp in split_cifar100 split_cifar100_20 permuted_mnist; do
            echo "Running HPO for $exp..."
            python -m experiments.hpo \
                --experiment $exp \
                --trials 50 \
                --output-dir "$OUT_DIR/hpo"
        done
        ;;

    m3_7)
        echo "=== M3.7: Generate paper artifacts ==="
        python -m experiments.comprehensive_eval --plot-only --results-dir "$OUT_DIR" --plots-dir "$PLOTS_DIR"
        python -m experiments.report --results-dir "$OUT_DIR" --plots-dir "$PLOTS_DIR"
        python -m experiments.profiling --results-dir "$OUT_DIR" --plots-dir "$PLOTS_DIR"
        ;;

    all)
        echo "=== FULL VALIDATION MATRIX ==="
        echo "This will take many hours. Running stages sequentially..."
        
        echo ""
        echo "Stage 1/5: M3.2 - Baseline LeanNGS"
        $0 m3_2
        
        echo ""
        echo "Stage 2/5: M3.3 - MNGS Profiles (param-matched)"
        $0 m3_3
        
        echo ""
        echo "Stage 3/5: M3.4 - Baselines"
        $0 m3_4
        
        echo ""
        echo "Stage 4/5: M3.3 LoRA - MNGS Profiles (LoRA efficient)"
        $0 m3_3_lora
        
        echo ""
        echo "Stage 5/5: M3.5-3.7 - Analysis & Artifacts"
        $0 m3_5
        $0 m3_6
        $0 m3_7
        
        echo ""
        echo "=== VALIDATION COMPLETE ==="
        echo "Results in: $OUT_DIR"
        echo "Plots in: $PLOTS_DIR"
        ;;

    *)
        echo "Usage: $0 [smoke|param_matched|m3_2|m3_3|m3_3_lora|m3_4|m3_5|m3_6|m3_7|all]"
        exit 1
        ;;
esac