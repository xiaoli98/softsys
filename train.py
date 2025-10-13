import lightning as L
from lightning.pytorch.loggers import WandbLogger
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray import tune
from datamodule import ImageDataModule
from lit_system import LitClassifier
import ray
from ray.tune import CLIReporter
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler


EPOCHS = 100
TOTAL_SAMPLES = 10  # Total hyperparameter configurations to try
CONCURRENT_TRIALS = 1
PARAM_GRID = {
    "model": tune.choice(['cnn', 'vit']),
    "learning_rate": tune.choice([1e-3, 1e-4]),
    "batch_size": tune.choice([16, 32]),
    # "epochs": tune.choice([10, 50, 100]),
    
    "model_depth": tune.choice([2, 3, 4]), #CNN only
    "kernel_size": tune.choice([3, 5]), #CNN only
    "latent_dim": tune.choice([16, 32, 64]), #CNN only
    "stride": tune.choice([1]), #CNN only
    "padding": tune.choice(['same']), #CNN only
}

DATA_DIR = "/data/malio/softsys/imgs_data" # absolute path to avoid ray issues
IMAGE_SIZE = (660, 4800)    # All images will be resized/padded to this size, (660, 4800) is the original size
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

def train_model(config):
    datamodule = ImageDataModule(
        data_dir=DATA_DIR,
        batch_size=config['batch_size'],
        image_size=IMAGE_SIZE
    )
    datamodule.setup(verbose=True) # for num classes
    
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
    
    logger = WandbLogger(project=WANDB_PROJECT_NAME, config=config, log_model='all')
    
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator='auto',
        devices="auto",
        log_every_n_steps=10,
        logger=logger,
        enable_progress_bar=True,
        callbacks=[TuneReportCheckpointCallback(
            {
                "loss": "val_loss",
                "mean_accuracy": "val_accuracy",
                "f1_score": "val_f1_score"
            },
            save_checkpoints=False,
            on="validation_end"
        )]
    )
    trainer.fit(lit_model, datamodule)

if __name__ == '__main__':
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "excludes": [
                "*.h5",
                "*.hdf5",
                # "*.pkl",
                "*.pth",
                "*.ckpt",
                "*.zip",
                "*.tar",
                "*.tar.gz",
                "data/",
                "datasets/",
                "models/",
                "checkpoints/",
                "logs/",
                "tmp/",
                "ray_results/",
                "wandb/",
                ".git/",
                "__pycache__/",
                "*.pyc",
                ".ipynb_checkpoints/",
                "*.mp4",
                "*.avi",
                "*.mov",
            ]
        })
    
    scheduler = ASHAScheduler(max_t=EPOCHS,
                              grace_period=1,
                              reduction_factor=2,
                              metric="f1_score",
                              mode="max")

    train_fn_with_resources = tune.with_resources(train_model, 
                                                  resources={"CPU": 1, "GPU": 1})
    
    search_alg = OptunaSearch(
        metric="f1_score",
        mode="max"
    )

    tuner = tune.Tuner(
        train_fn_with_resources,
        param_space=PARAM_GRID,
        tune_config=tune.TuneConfig(
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=TOTAL_SAMPLES,
            max_concurrent_trials=CONCURRENT_TRIALS,
        ),
        run_config=tune.RunConfig(
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="f1_score",
                checkpoint_score_order="max",
            ),
            name="softsys_image_classification",
            storage_path="/data/malio/ray_tmp/ray_results",
        ),
    )
    results = tuner.fit()
    print("Best hyperparameters found were:")
    print(results.get_best_results(metric="f1_score", mode="max"))