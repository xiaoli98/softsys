import torch
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, f1_score
from models import ConfigurableCNN, VisionTransformer

class LitClassifier(pl.LightningModule):
    def __init__(self, model_config, num_classes, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        if self.hparams.model_config['name'] == 'cnn':
            self.model = ConfigurableCNN(num_classes=num_classes, **self.hparams.model_config['params'])
        elif self.hparams.model_config['name'] == 'vit':
            self.model = VisionTransformer(num_classes=num_classes, **self.hparams.model_config['params'])
        else:
            raise ValueError(f"Unknown model name: {self.hparams.model_config['name']}")
            
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx, stage):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, labels, task='multiclass', num_classes=self.num_classes)
        f1 = f1_score(preds, labels, task='multiclass', num_classes=self.num_classes)
        
        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_f1_score', f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer