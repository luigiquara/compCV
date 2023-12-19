import copy
import torch
from torch import nn

# You can either pass a list of layers_id, i.e. the names of layers in model
# Or a list of (layer_id, layer) tuples
class SaveFeatures(nn.Module):
    def __init__(self, model, layers_id=None, layers=None):
        super().__init__()
        self.model = model

        if layers:
            self.layers_id = list(zip(*layers))[0]
            self.layers = list(zip(*layers))[1]

        elif layers_id:
            self.layers_id = layers_id
            self.layers = []
            for id in self.layers_id:
                self.layers.append(dict([*self.model.named_modules()])[id])

        self.features = {l_id: torch.empty(0) for l_id in self.layers_id}

        for id, layer in zip(self.layers_id, self.layers):
            layer.register_forward_hook(self.save_hook(id))

    def forward(self, x):
        out = self.model(x)
        return copy.deepcopy(self.features)

    def save_hook(self, l_id):
        def hook(model, input, output):
            self.features[l_id] = output
        return hook
