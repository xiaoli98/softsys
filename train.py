import itertools
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from datamodule import ImageDataModule
from lit_system import LitClassifier

PARAM_GRID = {
    'model': ['cnn', 'vit'],
    'learning_rate': [1e-3, 1e-4],
    'batch_size': [16, 32],
    'epochs': [10, 50, 100],
    'model_depth': [2, 3, 4],           # CNN only
    'kernel_size': [3, 5],           # CNN only
    'latent_dim': [64, 128, 256],        # CNN only

    'stride': [1],                   # CNN only
    'padding': ['same'],             # CNN only
}

DATA_DIR = "./imgs_data"
IMAGE_SIZE = 660    # All images will be resized/padded to this size
WANDB_PROJECT_NAME = "SoftSystem_Image_Grid_Search"

def show_images(datamodule):
    import matplotlib.pyplot as plt
    tl = datamodule.train_dataloader()
    for t in tl:
        images, labels = t  # unpack batch
        for idx, (img, label) in enumerate(zip(images, labels)):
            img = img.detach().cpu()
            # Normalize to 0-1 for display if needed
            img = img - img.min()
            if img.max() > 0:
                img = img / img.max()
            img = img.permute(1, 2, 0).numpy()  # CHW -> HWC

            plt.figure()
            plt.imshow(img)
            plt.title(f"Label: {label.item()}")
            plt.axis('off')

            # Build a unique filename
            fname = f"sample_image_{label.item()}_{idx}.png"
            plt.savefig(fname, bbox_inches='tight', pad_inches=0)
            print(f"Saved sample image to {fname}")
            plt.show()
    input("Press Enter to continue...")

def main():
    # TODO use wandb sweep for more advanced control
    keys, values = zip(*PARAM_GRID.items())
    run_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"--- Starting Grid Search with {len(run_configs)} combinations ---")

    for i, config in enumerate(run_configs):
        # Skip combinations that are not applicable (e.g., CNN params for ViT model)
        if config['model'] == 'vit' and any(k in config for k in ['model_depth', 'kernel_size', 'latent_dim']):
            # A simple way to avoid redundant ViT runs. 
            # We will run ViT only once per lr/batch_size/epochs combination.
            if config['model_depth'] != PARAM_GRID['model_depth'][0] or \
                config['kernel_size'] != PARAM_GRID['kernel_size'][0] or \
                config['latent_dim'] != PARAM_GRID['latent_dim'][0]:
                print(f"Skipping redundant ViT config: {config}")
                continue
        
        print(f"\n--- RUN {i+1}/{len(run_configs)} ---")
        print(f"Config: {config}")

        wandb_logger = WandbLogger(
            project=WANDB_PROJECT_NAME,
            config=config,
            job_type='train',
            group=f"model_{config['model']}"
        )
        
        datamodule = ImageDataModule(
            data_dir=DATA_DIR,
            batch_size=config['batch_size'],
            image_size=IMAGE_SIZE
        )
        datamodule.setup() # for num classes
        # show_images(datamodule) # Uncomment to visualize some images
        
        if config['model'] == 'cnn':
            model_config = {
                'name': 'cnn',
                'params': {
                    'img_size': IMAGE_SIZE,
                    'depth': config['model_depth'],
                    'kernel_size': config['kernel_size'],
                    'stride': config['stride'],
                    'padding': config['padding'],
                    'latent_dim': config['latent_dim']
                }
            }
        elif config['model'] == 'vit':
            model_config = {
                'name': 'vit',
                'params': {
                    'pretrained': True # Using pretrained ViT
                }
            }

        lit_model = LitClassifier(
            model_config=model_config,
            num_classes=datamodule.num_classes,
            learning_rate=config['learning_rate']
        )
        
        trainer = pl.Trainer(
            max_epochs=config['epochs'],
            logger=wandb_logger,
            accelerator='auto',
            devices=[0],
            log_every_n_steps=10,
            callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=10)]
        )

        trainer.fit(lit_model, datamodule)
        trainer.test(lit_model, datamodule)
        
        wandb.finish()

if __name__ == '__main__':
    main()