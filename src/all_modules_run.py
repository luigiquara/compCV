import argparse
import wandb

import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from module_extraction import get_modules_features

# putting these variables to global to pass them to all_modules_run
# to work with wandb sweeps, the function must be without arguments
global model
global tr_features
global val_features
global tr_target
global val_target

# module selection strategy
# 1. sort the pooled activations and pick the top-k ones, or
# 2. pick k modules at random
def module_selection(tr_features, val_features, selection_strategy, n_modules):
    # select at random n_modules
    if selection_strategy == 'random':
        tr_idxs = torch.randperm(tr_features.shape[1])[:n_modules]
        val_idxs = torch.randperm(val_features.shape[1])[:n_modules]

        training_modules = tr_features[:,tr_idxs]
        val_modules = val_features[:,val_idxs]

    # select the modules with the highest activation
    elif selection_strategy == 'topk':
        training_modules, training_idx = torch.topk(tr_features, n_modules)
        val_modules, test_idx = torch.topk(val_features, n_modules)

    return training_modules, val_modules

# fit the knn and get the accuracy on validation
def knn_fit(n_neighbors,weights, tr_modules, tr_target, val_modules, val_target):
    # fit the classifier on training set
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    knn.fit(tr_modules.numpy(), tr_target.numpy()) 

    # compute directly the accuracy
    # no need to compute the predictions beforehand
    val_accuracy = knn.score(val_modules, val_target)

    return val_accuracy

# get the modules activations
# try the composition of all modules
#def all_modules_run(model, tr_features, val_features, tr_target, val_target):
def all_modules_run():
    wandb.init()

    tr_modules, val_modules = module_selection(
        tr_features = tr_features[wandb.config.pooling],
        val_features = val_features[wandb.config.pooling],
        selection_strategy = wandb.config.module_choice,
        n_modules = wandb.config.n_modules
    )

    val_accuracy = knn_fit(
        n_neighbors = wandb.config.n_neighbors,
        weights = wandb.config.weights,
        tr_modules=tr_modules, val_modules=val_modules,
        tr_target=tr_target, val_target=val_target
    )

    wandb.log({'model': model, 'val_accuracy': val_accuracy})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str
    )
    flags = parser.parse_args()
    model = flags.model

    # define the parameters for grid search
    # and register them on wandb
    WANDB_PROJECT_NAME = 'CV & Compositionality'
    sweep_conf = {
        'method': 'grid',
        #'name': 'sweep',
        'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
        'parameters': {
            # knn hyperparameters
            'n_neighbors': {'values': [30, 40, 50, 70, 100, 200]},
            'weights': {'values': ['uniform', 'distance']},

            # module selection hyperparameters
            'module_choice': {'values': ['random', 'topk']},
            'pooling': {'values': ['max_pool', 'avg_pool']},
            'n_modules': {'values': [800, 1000, 1200, 1500]}
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_conf, project=WANDB_PROJECT_NAME)

    tr_features, val_features, tr_target, val_target = get_modules_features(model)
    wandb.agent(sweep_id, function=all_modules_run)
