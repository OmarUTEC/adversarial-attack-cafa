#!/bin/bash
# Ataques adversariales sobre modelos entrenados con SMOTE — AMLworld HI-Small
set -e
source cafa/bin/activate

run_attack() {
    MODEL=$1
    ATTACK=$2
    MODEL_PATH=$3
    OUTDIR="outputs/amlworld_hi_smote/${ATTACK}_${MODEL}"
    echo ""
    echo "=========================================="
    echo ">>> $(date '+%H:%M:%S')  $ATTACK  on  $MODEL (SMOTE)"
    echo "=========================================="
    python attack.py \
        data=amlworld_hi \
        ml_model=$MODEL \
        attack=$ATTACK \
        "ml_model.model_artifact_path=$MODEL_PATH" \
        "hydra.run.dir=$OUTDIR" \
        2>&1 | tee "${OUTDIR}_run.log"
    echo ">>> DONE: $ATTACK / $MODEL at $(date '+%H:%M:%S')"
}

# ── Logistic Regression ────────────────────────────────────
run_attack logistic_regression cafa           trained-models/amlworld_hi-logistic_regression-smote.ckpt
run_attack logistic_regression hop_skip_jump  trained-models/amlworld_hi-logistic_regression-smote.ckpt
run_attack logistic_regression boundary_attack trained-models/amlworld_hi-logistic_regression-smote.ckpt
run_attack logistic_regression square_attack  trained-models/amlworld_hi-logistic_regression-smote.ckpt

# ── XGBoost ────────────────────────────────────────────────
run_attack xgboost hop_skip_jump   trained-models/amlworld_hi-xgboost-smote
run_attack xgboost boundary_attack trained-models/amlworld_hi-xgboost-smote
run_attack xgboost square_attack   trained-models/amlworld_hi-xgboost-smote

# ── LSTM-Attention ─────────────────────────────────────────
run_attack lstm_attention cafa            trained-models/amlworld_hi-lstm_attention-smote.ckpt
run_attack lstm_attention hop_skip_jump   trained-models/amlworld_hi-lstm_attention-smote.ckpt
run_attack lstm_attention boundary_attack trained-models/amlworld_hi-lstm_attention-smote.ckpt
run_attack lstm_attention square_attack   trained-models/amlworld_hi-lstm_attention-smote.ckpt

echo ""
echo "=========================================="
echo "TODOS LOS ATAQUES SMOTE COMPLETADOS"
echo "=========================================="
