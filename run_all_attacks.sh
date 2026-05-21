#!/bin/bash
# Runs all compatible attacks on logistic_regression and lstm_attention
# Output dirs: outputs/{attack}_{model}

set -e
source cafa/bin/activate

run_attack() {
    MODEL=$1
    ATTACK=$2
    OUTDIR="outputs/${ATTACK}_${MODEL}"
    LOGFILE="outputs/${ATTACK}_${MODEL}_run.log"
    echo ""
    echo "=========================================="
    echo ">>> $(date '+%H:%M:%S')  $ATTACK  on  $MODEL"
    echo "=========================================="
    python attack.py \
        data=creditcard \
        ml_model=$MODEL \
        attack=$ATTACK \
        "hydra.run.dir=$OUTDIR" \
        2>&1 | tee "$LOGFILE"
    echo ">>> DONE: $ATTACK / $MODEL  at $(date '+%H:%M:%S')"
}

# ── Logistic Regression ────────────────────────────────────────────
run_attack logistic_regression cafa
run_attack logistic_regression hop_skip_jump
run_attack logistic_regression boundary_attack
run_attack logistic_regression simba
run_attack logistic_regression square_attack

# ── LSTM-Attention ─────────────────────────────────────────────────
run_attack lstm_attention cafa
run_attack lstm_attention hop_skip_jump
run_attack lstm_attention boundary_attack
run_attack lstm_attention simba
run_attack lstm_attention square_attack

echo ""
echo "=========================================="
echo "ALL ATTACKS COMPLETED at $(date '+%H:%M:%S')"
echo "=========================================="
