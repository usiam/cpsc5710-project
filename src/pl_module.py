from typing import Optional
import torch
import torch.nn as nn
from torchvision import transforms, models

import pytorch_lightning as pl
import torchmetrics
from torchmetrics import MetricCollection
# -----------------------
# LightningModule
# -----------------------
class LitResNet(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        pretrained: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()


        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        if pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        

        self.criterion = nn.CrossEntropyLoss()

        # ---- Metric Collections (train + val) ----
        common_metrics = {
            "Accuracy": torchmetrics.Accuracy(
                task="multiclass",
                num_classes=num_classes,
                average="micro",
            ),
            "Precision": torchmetrics.Precision(
                task="multiclass",
                num_classes=num_classes,
                average="macro",
            ),
            "Recall": torchmetrics.Recall(
                task="multiclass",
                num_classes=num_classes,
                average="macro",
            ),
            "F1Score": torchmetrics.F1Score(
                task="multiclass",
                num_classes=num_classes,
                average="macro",
            ),
            "AveragePrecision": torchmetrics.AveragePrecision(
                task="multiclass",
                num_classes=num_classes,
                average="macro",
            ),
        }

        metrics = MetricCollection(common_metrics)
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # 1. UPDATE: Just call the collection to update internal state. 
        # Do not assign the result to a variable for logging.
        self.train_metrics(logits, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # 2. LOG OBJECTS: Iterate over the collection and log the METRIC OBJECT
        # Lightning detects that 'metric' is a TorchMetric and handles global accumulation
        for name, metric in self.train_metrics.items():
            self.log(name, metric, on_step=False, on_epoch=True, 
                    prog_bar=("Accuracy" in name or "F1Score" in name))

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        self.val_metrics(logits, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        for name, value in self.val_metrics.items():
            # e.g. "val_Accuracy", "val_Precision", ...
            self.log(name, value, on_step=False, on_epoch=True,
                     prog_bar=("Accuracy" in name or "F1Score" in name))

    def configure_optimizers(self):
            # 1. CRITICAL: Filter parameters. 
            # Only pass parameters that are NOT frozen.
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()), # <--- CHANGED
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )

            # 2. RECOMMENDED: Switch to a reactive scheduler
            # This reduces LR when 'val_loss' stops going down
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=3, # Wait 3 epochs with no improvement before dropping LR
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss", # Watch val_loss, not F1
                    "frequency": 1
                },
            }
