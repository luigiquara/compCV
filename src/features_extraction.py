import time
from tqdm import tqdm
from collections import defaultdict
import torch

from save_features import Extractor

def get_relus(model):
    return [(name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.ReLU)]

# TODO: see what feature.shape[3] is and give a more meaningful name
# apply the pooling to all modules head
# both max and avg pooling
# then concat all the pooled activations
def pooling_and_concat(features):
    avg_pooled = []
    max_pooled = []

    # pooling of features
    # if kernel_size == matrix_size, it returns a single value
    for key, feature in features.items():
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
# those activation are pooled (both avg and max) and the concatenated
# TODO: output shape = ()
def extract_features(model, tr_loader, test_loader, device):
    extractor = Extractor(model, layers=get_relus(model))
    extractor.to(device)
    model.eval()

    tr_features = defaultdict(list)
    test_features = defaultdict(list)
    tr_target = []
    test_target = []
    # forward pass on training set
    with torch.no_grad():
        for mb, target in tqdm(tr_loader):
            mb = mb.to(device)

            start = time.time()
            features = extractor(mb)
            #print(f'Forward pass: {time.time()-start} seconds')

            start = time.time()
            for layer, v in features.items():
                tr_features[layer].append(v)
            #print(f'Torch.cat: {time.time()-start} seconds')
            tr_target.append(target)

        for layer, v in tr_features.items():
            tr_features[layer] = torch.cat(v)
        tr_target = torch.cat(tr_target)

        for mb, target in tqdm(test_loader):
            mb = mb.to(device)
            features = extractor(mb)

            for layer, v in features.items():
                test_features[layer].append(v)
            test_target.append(target)

        for layer, v in test_features.items():
            test_features[layer] = torch.cat(v)
        test_target = torch.cat(test_target)

    avg_tr_features, max_tr_features = pooling_and_concat(tr_features) 
    avg_test_features, max_test_features = pooling_and_concat(test_features)

    tr_features = {'avg_pool': avg_tr_features, 'max_pool': max_tr_features, 'target': tr_target}
    test_features = {'avg_pool': avg_test_features, 'max_pool': max_test_features, 'target': test_target}
    return tr_features, test_features