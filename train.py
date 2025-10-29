import lightning as L
import yaml
import argparse
from lightning.pytorch.loggers import WandbLogger
from ray import tune
from datamodule import ImageDataModule
from lit_system import LitClassifier
import ray
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

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

def train_model(config, data_config, epochs, data_dir, pooling_size, wandb_project_name, run_name):
    data_config = data_config
    datamodule = ImageDataModule(
        data_dir=data_dir,
        batch_size=config['batch_size'],
        # image_size=tuple(image_size),
        sobel=data_config.get('sobel', False),
    )
    datamodule.setup(verbose=True, contrastive=config.get('use_contrastive', False)) # for num classes
    
    if config['model'] == 'cnn':
        model_config = {
            'name': 'cnn',
            'params': {
                'pooling_size': tuple(pooling_size),
                'depth': config['model_depth'],
                'kernel_size': config['kernel_size'],
                'stride': config['stride'],
                'padding': config['padding'],
                'latent_dim': config['latent_dim'],
            }
        }
    elif config['model'] == 'vit':
        model_config = {
            'name': 'vit',
            'params': {
                'pretrained': True, # Using pretrained ViT
            }
        }

    lit_model = LitClassifier(
        model_config=model_config,
        num_classes=datamodule.num_classes,
        learning_rate=config['learning_rate'],
        use_contrastive=config.get('use_contrastive', False),
        supcon_weight=config.get('supcon_weight', 0.5)
    )

    logger = WandbLogger(project=wandb_project_name, name=run_name, config=config, log_model=True)

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator='auto',
        devices="auto",
        log_every_n_steps=10,
        logger=logger,
        enable_progress_bar=True,
        callbacks=[
            TuneReportCheckpointCallback(
                {
                    "loss": "val_loss",
                    "mean_accuracy": "val_accuracy",
                    "f1_score": "val_f1_score"
                },
                save_checkpoints=False,
                on="validation_end"
            ),
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath="/data/malio/softsys/checkpoints",
                filename=f"model-{config['model']}-" + "{epoch:02d}-{val_loss:.2f}",
                monitor="val_loss",
                mode="min",
                save_top_k=2,
                save_last=False 
            ),
            ]
    )
    trainer.fit(lit_model, datamodule)
    wandb.finish()

def main(config):
    # Construct the search space for Ray Tune from the config file
    param_grid = {}
    for key, value in config['param_grid'].items():
        if value['tune_type'] == 'choice':
            param_grid[key] = tune.choice(value['values'])
        elif value['tune_type'] == 'uniform':
            param_grid[key] = tune.uniform(value['min'], value['max'])

    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "excludes": [ # some default excludes
                "*.h5", "*.hdf5", "*.pth", "*.ckpt", "*.zip", "*.tar", "*.tar.gz",
                "data/", "datasets/", "models/", "checkpoints/", "logs/", "tmp/",
                "ray_results/", "wandb/", ".git/", "__pycache__/", "*.pyc",
                ".ipynb_checkpoints/", "*.mp4", "*.avi", "*.mov",
            ]
        })
    
    scheduler = ASHAScheduler(max_t=config['epochs'],
                              grace_period=5,
                              reduction_factor=2,
                              metric="f1_score",
                              mode="max")

    trainable_with_params = tune.with_parameters(
        train_model,
        data_config=config.get('dataset', {}),
        epochs=config['epochs'],
        data_dir=config['data_dir'],
        # image_size=config['image_size'],
        pooling_size=config['pooling_size'],
        wandb_project_name=config['wandb_project_name'],
        run_name=config['run_name']
    )

    train_fn_with_resources = tune.with_resources(trainable_with_params, 
                                                  resources={"CPU": 1, "GPU": 1})
    
    search_alg = OptunaSearch(
        metric="f1_score",
        mode="max"
    )

    tuner = tune.Tuner(
        train_fn_with_resources,
        param_space=param_grid,
        tune_config=tune.TuneConfig(
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=config['total_samples'],
            max_concurrent_trials=config['concurrent_trials'],
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
    print(results.get_best_result(metric="f1_score", mode="max"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning for image classification.")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration YAML file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)