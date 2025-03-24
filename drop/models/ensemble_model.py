import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate
import logging

class EnsembleModel(pl.LightningModule):
    def __init__(self, folds_used, model_paths, model_config):
        super().__init__()
        self.folds_used = folds_used
        self.models = nn.ModuleList()  # Wrap models in a ModuleList
        self.num_models = len(self.folds_used)
        if self.num_models != len(model_paths):
            logging.warning("Ensemble model constructed with subset of original folds.")

        # Load only the models corresponding to the used folds
        for fold in self.folds_used:
            model = instantiate(model_config)
            model.load_state_dict(self.get_state_dict(model_paths[fold]))
            self.models.append(model)

    def forward(self, x, extra_features=None):
        outputs_list = []

        for fold in range(len(self.folds_used)):
            if extra_features is not None:
                outputs, _ = self.models[fold](x, extra_features)
            else:
                outputs, _ = self.models[fold](x)
            outputs_list.append(outputs)

        outputs_ensemble = torch.stack(outputs_list).mean(0)

        return outputs_ensemble, None

    def get_state_dict(self, model_path):
        state_dict = torch.load(model_path)["state_dict"]
        adapted_state_dict = {k.split("model.")[-1]: v for k, v in state_dict.items()}
        return adapted_state_dict