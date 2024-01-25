import argparse
import wandb

import torch
from torch import nn
from sklearn.linear_model import LogisticRegression

from utils import load_cub, get_model
from features_extraction import forward_pass_extraction, pooling_and_concat


# take a given number of subtrees
# put a logistic regressor on top to classify
# the maximum number of modules is 17 in resnet50
def logReg(model):
    device = torch.device('cuda')
    resnet = get_model(model)
    tr_loader, test_loader = load_cub(batch_size=32)
    tr_features, tr_target, test_features, test_target = forward_pass_extraction(resnet, tr_loader, test_loader, device)

    for n_modules in range(17, 0, -1):
        print(f'Using {n_modules} modules')
        modules = list(tr_features.keys())[:n_modules]

        # get only the activations from selected modules
        # pool and concat them
        partial_tr_features = {layer: tr_features[layer] for layer in modules}
        partial_test_features = {layer: test_features[layer] for layer in modules}
        tr_pool_concat = pooling_and_concat(partial_tr_features)
        test_pool_concat = pooling_and_concat(partial_test_features)
        partial_tr_features = {'avg_pool': tr_pool_concat[0], 'max_pool': tr_pool_concat[1]}
        partial_test_features = {'avg_pool': test_pool_concat[0], 'max_pool': test_pool_concat[1]}

        acc = {'avg_pool': 0, 'max_pool': 0}

        # use logistic regression as the classification head
        for pooling_method in ['avg_pool', 'max_pool']:
            classifier = LogisticRegression(max_iter=400, n_jobs=-1)
            classifier.fit(partial_tr_features[pooling_method].numpy(), tr_target.numpy())
            print('Fitted')

            acc[pooling_method] = classifier.score(partial_test_features[pooling_method].numpy(), test_target.numpy())
            print('Score computed')
            print(f'acc: {acc}')
        
        wandb.log({'accuracy_avg_pool': acc['avg_pool'], 'accuracy_max_pool': acc['max_pool'], 'n_modules': n_modules})
    
    wandb.config.update({
        'model': model,
        'batch_size': 32
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str
    )
    flags = parser.parse_args()

    wandb.init(project = 'CV & Compositionality')
    logReg(flags.model)