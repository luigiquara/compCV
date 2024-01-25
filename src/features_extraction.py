import time
from tqdm import tqdm
from collections import defaultdict
import torch

from save_features import Extractor

def get_relus(model):
    return [(name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.ReLU)]

# apply the pooling to all modules head
# both max and avg pooling
# then concat all the pooled activations
def pooling_and_concat(features):
    avg_pooled = []
    max_pooled = []

    # pooling of features
    # if kernel_size == matrix_size, it returns a single value
    for key, feature in features.items():
        feature = torch.cat(feature) # concat features from different batches
        assert feature.shape[2] == feature.shape[3], feature.shape
        matrix_size = feature.shape[3]
        avg_layer = torch.nn.AvgPool2d(kernel_size = matrix_size)
        max_layer = torch.nn.MaxPool2d(kernel_size = matrix_size)

        avg_pooled.append(torch.squeeze(avg_layer(feature)))
        max_pooled.append(torch.squeeze(max_layer(feature)))

    avg_pooled = torch.cat(avg_pooled, dim=1)
    max_pooled = torch.cat(max_pooled, dim=1)
    
    return avg_pooled, max_pooled


# return the activations from all relu layers
def forward_pass_extraction(model, tr_loader, test_loader, device):
    extractor = Extractor(model, layers=get_relus(model))
    extractor.to(device)
    model.eval()

    tr_features = defaultdict(list)
    test_features = defaultdict(list)
    tr_target = []
    test_target = []

    with torch.no_grad():

        # forward pass on training set
        for mb, target in tqdm(tr_loader):
            mb = mb.to(device)

            features = extractor(mb)
            for layer, v in features.items():
                tr_features[layer].append(v)
            tr_target.append(target)

        # forward pass on test set
        for mb, target in tqdm(test_loader):
            mb = mb.to(device)

            features = extractor(mb)

            for layer, v in features.items():
                test_features[layer].append(v)
            test_target.append(target)

        tr_target = torch.cat(tr_target)
        test_target = torch.cat(test_target)
    
    return tr_features, tr_target, test_features, test_target