import os
from typing import List

import torch

from train_models import LResNet

class ModelLoader():
    def __init__(self, base_path):
        self.base_path = base_path

    # kwargs can contain:
    # n_models = the number of models to take, when not all
    def load_models(self, strategy: str, **kwargs) -> List[torch.nn.Module]:
        if strategy == 'all': models = self._all_models()
        elif strategy == 'random': models = self._random_models(kwargs['n_models'])

        return models

    def _all_models(self) -> List[torch.nn.Module]:
        models = []
        for idx in range(10): # number of experiences
            m = LResNet.load_from_checkpoint(self._get_path(idx))
            m = self._remove_classifier(m)
            models.append(m)
        return models

    def _random_models(self, n_models: int) -> List[torch.nn.Module]:
        raise NotImplementedError

    def _get_path(self, exp_id: int) -> str:
        path = self.base_path + f'exp_{exp_id}/checkpoints/'
        ckpt = os.listdir(path)[0]
        return path + ckpt

    def _remove_classifier(self, model: torch.nn.Module) -> torch.nn.Module:
        layers = list(model.resnet.children())[:-1] #remove loss
        feature_extractor = torch.nn.Sequential(*layers)

        return feature_extractor