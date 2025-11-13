import logging
import os
import shutil
from typing import Any, Dict, List

import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score
import torch
from torch import nn

from src.datasets.load_tabular_data import TabularDataset

logger = logging.getLogger(__name__)


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LitLogisticRegression(pl.LightningModule):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            lr: float = 1e-3,
            weight_decay: float = 1e-5,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = LogisticRegressionModel(input_dim, output_dim)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = logits.argmax(dim=1, keepdim=True)
        acc = preds.eq(y.view_as(preds)).float().mean()
        try:
            auc = roc_auc_score(y.cpu(), preds.cpu())
        except ValueError:
            auc = torch.tensor(0.0)
        self.log(f"{stage}_loss", loss)
        self.log(f"{stage}_acc", acc)
        self.log(f"{stage}_auc", auc)
        self.log(f"{stage}_hp_metric", auc, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

    @classmethod
    def define_trial_parameters(cls, trial: optuna.trial.Trial) -> Dict[str, Any]:
        return dict(
            lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        )


def train(
        hyperparameters: Dict[str, Any],
        trainset: torch.utils.data.Dataset,
        testset: torch.utils.data.Dataset,
        tab_dataset: TabularDataset,
        model_artifact_path: str = None,
        additional_callbacks: List[pl.callbacks.Callback] = None
):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4096, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4096, shuffle=False)

    model = LitLogisticRegression(
        input_dim=tab_dataset.n_features,
        output_dim=tab_dataset.n_classes,
        **hyperparameters
    )

    callbacks = [
        ModelCheckpoint(
            monitor="val_hp_metric",
            mode="max",
            filename='{epoch}-{val_hp_metric:.3f}'
        )
    ]
    if additional_callbacks:
        callbacks += additional_callbacks

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=callbacks,
        default_root_dir=f"outputs/training/logreg/{tab_dataset.data_parameters['dataset_name']}/",
    )
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=testloader)

    results = {
        'best_val_loss': trainer.callback_metrics['val_loss'].item(),
        'best_val_acc': trainer.callback_metrics['val_acc'].item(),
        'best_val_hp_metric': trainer.callback_metrics['val_hp_metric'].item(),
        'best_model_path': trainer.checkpoint_callback.best_model_path,
        'best_model_val_hp_metric': trainer.checkpoint_callback.best_model_score,
    }

    if model_artifact_path is not None:
        shutil.copy(trainer.checkpoint_callback.best_model_path, model_artifact_path)
        hparams_src = os.path.join(
            os.path.dirname(os.path.dirname(trainer.checkpoint_callback.best_model_path)),
            'hparams.yaml'
        )
        hparams_dst = os.path.join(
            os.path.dirname(model_artifact_path),
            f'{os.path.basename(model_artifact_path)}.hparams.yaml'
        )
        shutil.copy(hparams_src, hparams_dst)
        logger.info(f"Saved model's artifact to {model_artifact_path}")

    logger.info(f"Finished logistic regression training. Results: {results}")
    return results


def grid_search_hyperparameters(
        trainset: torch.utils.data.Dataset,
        testset: torch.utils.data.Dataset,
        tab_dataset: TabularDataset,
):
    def optuna_hpo_objective(trial: optuna.trial.Trial) -> float:
        hyperparameters = LitLogisticRegression.define_trial_parameters(trial)
        results = train(
            hyperparameters,
            trainset=trainset,
            testset=testset,
            tab_dataset=tab_dataset,
            additional_callbacks=[
                optuna.integration.PyTorchLightningPruningCallback(trial, monitor="val_hp_metric")
            ]
        )
        return results['best_model_val_hp_metric']

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(optuna_hpo_objective, n_trials=50, timeout=600)

    best_hparams = study.best_trial.params
    logger.info(f"Finished logistic regression HPO. Best hyperparameters: {best_hparams}")
    return best_hparams
