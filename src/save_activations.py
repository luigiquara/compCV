import torch
from torch import nn

class SaveFeatures(nn.Module):
    def __init__(self, model, layers_id):
        super().__init__()
        self.model = model
        self.layers_id = layers_id
        self.features = {l_id: torch.empty(0) for l_id in layers_id}

        for l_id in self.layers_id:
            layer = dict([*self.model.named_modules()])[l_id]
            layer.register_forward_hook(self.save_hook(l_id))

    def forward(self, x):
        out = self.model(x)
        return self.features

    def save_hook(self, l_id):
        def hook(model, input, output):
            self.features[l_id] = output
        return hook
