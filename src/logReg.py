import time
import random
import argparse
import wandb
import numpy as np

import torch
from torch import nn
from sklearn.linear_model import LogisticRegression

from utils import load_cub, get_model
from features_extraction import forward_pass_extraction, pooling_and_concat

# fit the logistic regressor
# get the accuracy on test set
def logReg(x_tr, y_tr, x_test, y_test):
    classifier = LogisticRegression(max_iter=100)

    start = time.time()
    classifier.fit(x_tr, y_tr)
    print(f'Fitted in {time.time() - start} seconds')

    acc = classifier.score(x_test, y_test)
    print('Score computed')
    print(f'acc: {acc}')
    
    return acc


# take a given number of subtrees
# put a logistic regressor on top to classify
# the maximum number of modules is 17 in resnet50
def run(model):
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
        #for pooling_method in ['avg_pool', 'max_pool']:
        # max_loop seems always >= than av_pool
        for pooling_method in ['max_pool']: #use the loop just to maintain compatibility
            classifier = LogisticRegression(max_iter=100)

            start = time.time()
            classifier.fit(partial_tr_features[pooling_method].numpy(), tr_target.numpy())
            print(f'Fitted in {time.time() - start} seconds')

            acc[pooling_method] = classifier.score(partial_test_features[pooling_method].numpy(), test_target.numpy())
            print('Score computed')
            print(f'acc: {acc}')
        
        wandb.log({'accuracy_avg_pool': acc['avg_pool'], 'accuracy_max_pool': acc['max_pool'], 'n_modules': n_modules})
    
    wandb.config.update({
        'model': model,
        'classifier': 'LogisticRegression',
        'batch_size': 32
    })

def run_random_modules(model):
    device = torch.device('cuda')
    resnet = get_model(model)
    tr_loader, test_loader = load_cub(batch_size=32)
    tr_features, tr_target, test_features, test_target = forward_pass_extraction(resnet, tr_loader, test_loader, device)

    # 3 broad number of modules to start experimenting
    n_modules = [5, 10, 15]
    n_tries = 5

    for n in n_modules:
        # repeat for a number of tries
        acc = {'max_pool': []}
        for i in range(n_tries):
            # select randomly the modules to use
            idx = random.sample(range(17), n)
            total_modules = list(tr_features.keys())
            modules = [total_modules[i] for i in idx]

            print(f'Try {i} with {n} modules')
            print(f'Using modules: {idx}')

            # retrieve the features from the selected modules
            partial_tr_features = {layer: tr_features[layer] for layer in modules}
            partial_test_features = {layer: test_features[layer] for layer in modules}
            # pooling and concat of the selected features
            tr_pool_concat = pooling_and_concat(partial_tr_features)
            test_pool_concat = pooling_and_concat(partial_test_features)
            partial_tr_features = {'avg_pool': tr_pool_concat[0], 'max_pool': tr_pool_concat[1]}
            partial_test_features = {'avg_pool': test_pool_concat[0], 'max_pool': test_pool_concat[1]}

            # fit the classifier and get the accuracy
            acc['max_pool'].append(logReg(partial_tr_features['max_pool'].numpy(), tr_target.numpy(),
                                     partial_test_features['max_pool'].numpy(), test_target.numpy()))
        wandb.log({'accuracy_max_pool': acc['max_pool'], 'n_modules': n})
        wandb.log({'mean_accuracy': np.mean(acc['max_pool']),'std_accuracy': np.std(acc['max_pool']), 'n_modules': n})

    wandb.config.update({
        'model': model,
        'classifier': 'LogisticRegression',
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
    run_random_modules(flags.model)