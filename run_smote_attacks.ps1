# Run all SMOTE attacks on trained models — AMLworld HI
# Output: outputs/amlworld_hi_smote/{attack}_{model}/

# Activate venv
& d:\TESIS-2\adversarial-attack-cafa\cafa\Scripts\Activate.ps1

function Run-Attack {
    param(
        [string]$Model,
        [string]$Attack,
        [string]$ModelPath
    )
    
    $outdir = "outputs/amlworld_hi_smote/${Attack}_${Model}"
    $logfile = "${outdir}_run.log"
    
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host ">>> $(Get-Date -Format 'HH:mm:ss')  $Attack  on  $Model (SMOTE)" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
    
    python attack.py `
        data=amlworld_hi `
        ml_model=$Model `
        attack=$Attack `
        "ml_model.model_artifact_path=$ModelPath" `
        "hydra.run.dir=$outdir" `
        2>&1 | Tee-Object -FilePath $logfile
    
    Write-Host ">>> DONE: $Attack / $Model at $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "  ATAQUES ADVERSARIALES SMOTE - AMLworld HI" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

# Logistic Regression
Run-Attack logistic_regression cafa           trained-models/amlworld_hi-logistic_regression-smote.ckpt
Run-Attack logistic_regression hop_skip_jump  trained-models/amlworld_hi-logistic_regression-smote.ckpt
Run-Attack logistic_regression boundary_attack trained-models/amlworld_hi-logistic_regression-smote.ckpt
Run-Attack logistic_regression square_attack  trained-models/amlworld_hi-logistic_regression-smote.ckpt

# XGBoost
Run-Attack xgboost hop_skip_jump   trained-models/amlworld_hi-xgboost-smote
Run-Attack xgboost boundary_attack trained-models/amlworld_hi-xgboost-smote
Run-Attack xgboost square_attack   trained-models/amlworld_hi-xgboost-smote

# LSTM-Attention
Run-Attack lstm_attention cafa            trained-models/amlworld_hi-lstm_attention-smote.ckpt
Run-Attack lstm_attention hop_skip_jump   trained-models/amlworld_hi-lstm_attention-smote.ckpt
Run-Attack lstm_attention boundary_attack trained-models/amlworld_hi-lstm_attention-smote.ckpt
Run-Attack lstm_attention square_attack   trained-models/amlworld_hi-lstm_attention-smote.ckpt

Write-Host ""
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "  TODOS LOS ATAQUES SMOTE COMPLETADOS" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
