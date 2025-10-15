import torch
import torch.nn as nn
import lightning as L
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
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

        self.ce_loss = nn.BCELoss()
        if self.use_contrastive:
            self.supcon_loss = SupConLoss()

        self.train_f1 = BinaryF1Score()
        self.train_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.val_acc = BinaryAccuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.use_contrastive:
            # Contrastive learning path
            # Assumes batch is a tuple: ((view1, view2), labels)
            (images1, images2), labels = batch
            # print("image shapes:", images1.shape, images2.shape)
            images = torch.cat([images1, images2], dim=0)
            # print(f"Combined image shape: {images.shape}")
            logits, projections = self.model(images)
            
            bsz = labels.shape[0]
            projections = projections.view(bsz, 2, -1)
            logits1,logits2 = torch.split(logits, [bsz, bsz], dim=0)

            loss_ce = self.ce_loss(logits1, labels)
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
        batch_size = labels.shape[0]
        if self.use_contrastive:
            input_ = torch.cat((images[0], images[1]), dim=0)
            logits, projections = self.model(input_)
            projections = projections.view(batch_size, 2, -1)
            logits1,logits2 = torch.split(logits, [batch_size, batch_size], dim=0)
            
            loss_supcon = self.supcon_loss(projections, labels)
            loss_ce = self.ce_loss(logits1, labels)
            loss = (1 - self.supcon_weight) * loss_ce + self.supcon_weight * loss_supcon

            self.log('val_loss_ce', loss_ce, prog_bar=True)
            self.log('val_loss_supcon', loss_supcon, prog_bar=True)
            self.log('val_loss', loss, prog_bar=True)
            self.val_acc(logits1, labels)
            self.val_f1(logits1, labels)
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
    
if __name__ == "__main__":
    from datamodule import ImageDataModule
    from losses import SupConLoss
    # Example usage
    model_config = {
        'name': 'cnn',
        'params': {
            'img_size': (4800, 660),
            'depth': 2,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'latent_dim': 128,
        }
    }
    num_classes = 2
    learning_rate = 1e-3
    use_contrastive = True
    supcon_weight = 0.5

    lit_model = LitClassifier(
        model_config=model_config,
        num_classes=num_classes,
        learning_rate=learning_rate,
        use_contrastive=use_contrastive,
        supcon_weight=supcon_weight
    )

    datamodule = ImageDataModule(
        data_dir="/data/malio/softsys/imgs_data",
        batch_size=4,
        image_size=(4800, 660),
        sobel= False,
    )
    datamodule.setup(verbose=True, contrastive=True)
    tl = datamodule.train_dataloader()
    loss_fn = SupConLoss()
    acc_loss = BinaryAccuracy()
    f1_loss = BinaryF1Score()
    ce_loss = nn.BCELoss()
    batch_size = 4
    for batch in tl:
        samples, labels = batch
        print(f"sample len: {len(samples)}")
        for s in samples:
            print(s.shape, s.dtype)
        input_ = torch.cat((samples[0], samples[1]), dim=0)
        print(f'Input shape: {input_.shape}, dtype: {input_.dtype}')
        logits, projections = lit_model(input_)
        print("Logits shape:", logits.shape)
        print("Projections shape:", projections.shape)

        f1,f2 = torch.split(projections, [batch_size, batch_size], dim=0)
        l1, l2 = torch.split(logits, [batch_size, batch_size], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = loss_fn(features , labels)
        
        loss_ce = ce_loss(l1, labels)
        print(f"Cross-entropy loss: {loss_ce.item()}")
        print(f"SupCon loss: {loss_fn(features , labels).item()}")
        accuracy = acc_loss(l1, labels)
        f1_score = f1_loss(l1, labels)
        print(f"Accuracy: {accuracy.item()}")
        print(f"F1 Score: {f1_score.item()}")
        print(f"Loss: {loss.item()}")