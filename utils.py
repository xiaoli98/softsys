import lightning as L
import wandb
import torch
from ray import tune

class RayTuneReportCallback(L.Callback):
    def __init__(self, report_dict=None):
        super().__init__()
        self.report_dict = report_dict

    """Custom callback to report metrics to Ray Tune during training"""
    def on_validation_epoch_end(self, trainer, pl_module):
        # Get metrics from trainer
        metrics = trainer.callback_metrics
        
        report_dict = {}
        
        if self.report_dict is not None:
            for key, val in self.report_dict.items():
                if val in metrics:
                    if isinstance(metrics[val], torch.Tensor):
                        report_dict[key] = metrics[val].item()
                    else:
                        report_dict[key] = metrics[val]
        else:
            # If no specific report_dict, report all metrics
            for key, val in metrics.items():
                if isinstance(val, torch.Tensor):
                    val = val.item()
                report_dict[key] = val
        wandb.log(report_dict)
        tune.report(**report_dict)