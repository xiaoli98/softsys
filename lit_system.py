import torch
import torch.nn as nn
import lightning as L
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchmetrics.functional import accuracy, f1_score
from models import ConfigurableCNN, VisionTransformer
from losses import SupConLoss # Make sure losses.py exists

class LitClassifier(L.LightningModule):
    def __init__(self, model_config, num_classes, learning_rate=1e-3, use_contrastive=False, supcon_weight=0.5):
        super().__init__()
        self.save_hyperparameters()
        
        # Pass the use_contrastive flag to the model
        model_params = model_config.get('params', {})
        model_params['use_contrastive'] = use_contrastive

        if model_config['name'] == 'cnn':
            self.model = ConfigurableCNN(num_classes=num_classes, **model_params)
        elif model_config['name'] == 'vit':
            self.model = VisionTransformer(num_classes=num_classes, **model_params)
        
        self.learning_rate = learning_rate
        self.use_contrastive = use_contrastive
        self.supcon_weight = supcon_weight

        self.ce_loss = nn.CrossEntropyLoss()
        if self.use_contrastive:
            self.supcon_loss = SupConLoss()

        self.train_f1 = MulticlassF1Score(num_classes=num_classes)
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.use_contrastive:
            # Contrastive learning path
            # Assumes batch is a tuple: ((view1, view2), labels)
            (images1, images2), labels = batch
            images = torch.cat([images1, images2], dim=0)

            logits, projections = self.model(images)
            
            bsz = labels.shape[0]
            projections = projections.view(bsz, 2, -1)

            loss_ce = self.ce_loss(logits, torch.cat([labels, labels], dim=0))
            loss_supcon = self.supcon_loss(projections, labels)
            loss = (1 - self.supcon_weight) * loss_ce + self.supcon_weight * loss_supcon

            # Log contrastive-specific losses
            self.log('train_loss_ce', loss_ce)
            self.log('train_loss_supcon', loss_supcon)
            
            # Use one set of logits for standard metrics
            logits, _ = torch.split(logits, [bsz, bsz], dim=0)

        else:
            # Standard training path
            images, labels = batch
            logits = self.model(images)
            loss = self.ce_loss(logits, labels)

        # --- Common logic for logging and metrics ---
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_acc(logits, labels)
        self.train_f1(logits, labels)
        self.log('train_accuracy', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1_score', self.train_f1, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        if self.use_contrastive:
            logits, projections = self.model(images)
            loss_supcon = self.supcon_loss(projections, labels)
            loss_ce = self.ce_loss(logits, labels)
            loss = (1 - self.supcon_weight) * loss_ce + self.supcon_weight * loss_supcon

            self.log('val_loss_ce', loss_ce, prog_bar=True)
            self.log('val_loss_supcon', loss_supcon, prog_bar=True)
            self.log('val_loss', loss, prog_bar=True)
            self.val_acc(logits, labels)
            self.val_f1(logits, labels)
            self.log('val_accuracy', self.val_acc, prog_bar=True)
            self.log('val_f1_score', self.val_f1, prog_bar=True)
            return loss
        else:
            logits = self.model(images)
            loss = self.ce_loss(logits, labels)
        
            self.val_acc(logits, labels)
            self.val_f1(logits, labels)
            
            self.log('val_loss', loss, prog_bar=True)
            self.log('val_accuracy', self.val_acc, prog_bar=True)
            self.log('val_f1_score', self.val_f1, prog_bar=True)
            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer