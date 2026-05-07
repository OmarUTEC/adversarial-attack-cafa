import logging
import torch

from src.models.mlp import LitMLP
from src.models.lstm_attention import LitLSTMAttention
from src.models.logistic_regression import LitLogisticRegression
from src.models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)


def load_trained_model(saved_model_path, model_type='mlp',
                       dataset=None, eval_mode=True):
    """
    Loads the required model from the saved models directory
    """
    logger.info(f"Loading model from {saved_model_path}")

    _cls_map = {
        'mlp': LitMLP,
        'lstm_attention': LitLSTMAttention,
        'logistic_regression': LitLogisticRegression,
    }
    if model_type == 'xgboost':
        return XGBoostModel.load(saved_model_path)
    elif model_type in _cls_map:
        map_location = None if torch.cuda.is_available() else 'cpu'
        cls = _cls_map[model_type]
        try:
            model = cls.load_from_checkpoint(saved_model_path, map_location=map_location)
        except RuntimeError:
            # Checkpoint saved before register_buffer refactor — load with strict=False
            model = cls.load_from_checkpoint(saved_model_path, map_location=map_location, strict=False)
    else:
        raise NotImplementedError(f"Unknown model type: {model_type}")

    if eval_mode:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    return model
