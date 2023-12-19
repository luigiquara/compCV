import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset
from avalanche.benchmarks.datasets import CUB200

from torchvision.transforms import v2 as transforms
from torchvision import models
from sklearn.neighbors import KNeighborsClassifier

from save_activations import SaveFeatures

from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import accuracy_score

# load and return the selected type of resnet
def get_model(model):
    if model == 'resnet18':
        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT', progress=True)
    elif model == 'resnet34':
        resnet = models.resnet34(weights='ResNet34_Weights.DEFAULT', progress=True)
    elif model == 'resnet50':
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT', progress=True)
    elif model == 'resnet101':
        resnet = models.resnet101(weights='ResNet101_Weights.DEFAULT', progress=True)
    elif model == 'resnet152':
        resnet = models.resnet152(weights='ResNet152_Weights.DEFAULT', progress=True)
    return resnet

# return the list of string identifiers of each relu layer in the model
def get_relu_layers(model):
    return [(name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.ReLU)]

def pool_and_concat(features):
    # pooling of training features
    # if kernel_size == matrix_size, it returns a single value
    pooled = []
    for key, feature in features.items():
        assert feature.shape[2] == feature.shape[3], feature.shape
        avg_layer = torch.nn.AvgPool2d(kernel_size = feature.shape[3])

        # needed to change the view to apply pad_sequence
        # it needs the last dimension to be equal
        pooled.append(torch.squeeze(avg_layer(feature)).view((-1, feature.shape[0])))

    # concat the activations from all layers
    pooled = torch.cat(pooled).view((feature.shape[0], -1))
    return pooled


# Get the activation at each layer
# The modules are the subnets that end at every kernel at each layer
# Try the composition of all modules
def all_modules_run(model):

    # load and preprocess CUB dataset
    preprocessing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToImageTensor(),
        transforms.ConvertImageDtype(),
        # mean and std values from https://pytorch.org/hub/pytorch_vision_resnet/
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cub_training = CUB200(root='/home/lquarantiello/compCV', transform=preprocessing, download=False)
    cub_test = CUB200(root='/home/lquarantiello/compCV', transform=preprocessing, train=False, download=False)

    #cub_training = Subset(cub_training, torch.arange(0,300))
    #cub_test = Subset(cub_test, torch.arange(0,200))

    tr_loader = DataLoader(cub_training, batch_size=len(cub_training))
    test_loader = DataLoader(cub_test, batch_size=len(cub_test))

    # load the model & register the hook at the last layer before classification
    print(f'\n\nUsing {model}\n\n')
    resnet = get_model(model)
    
    resnet.eval()
    extractor = SaveFeatures(resnet, layers=get_relu_layers(resnet))

    with torch.no_grad():
        for batch, training_target in tqdm(tr_loader):
            training_features = extractor(batch)

        for batch, test_target in tqdm(test_loader):
            test_features = extractor(batch)

    pooled_training_features = pool_and_concat(training_features)
    pooled_test_features = pool_and_concat(test_features)

    # grid search on the knn
    # get the configuration with the best accuracy 
    p_grid = {
        'n_neighbors': [30, 40, 50, 70, 100, 200],
        'weights': ['uniform', 'distance'],
        'module_choice': ['random', 'topk'],
        'n_modules': [800, 1000, 1200, 1500]
    } 

    p_grid = ParameterGrid(p_grid) 

    best_acc = 0.0
    best_params = {}
    for params in p_grid:
        knn = KNeighborsClassifier(n_neighbors=params['n_neighbors'], weights=params['weights'])
        params['n_modules'] = min(params['n_modules'], pooled_training_features.shape[1])

        # select at random n_modules
        if params['module_choice'] == 'random':
            tr_idxs = torch.randperm(pooled_training_features.shape[1])[:params['n_modules']]
            test_idxs = torch.randperm(pooled_test_features.shape[1])[:params['n_modules']]

            training_modules = pooled_training_features[:,tr_idxs]
            test_modules = pooled_test_features[:,test_idxs]


        # select the modules with the highest activation
        elif params['module_choice'] == 'topk':
            training_modules, training_idx = torch.topk(pooled_training_features, params['n_modules'])
            test_modules, test_idx = torch.topk(pooled_test_features, params['n_modules'])

        # fit the classifier on training set
        knn.fit(training_modules.numpy(), training_target.numpy()) 

        # compute directly the accuracy
        # no need to compute the predictions beforehand
        acc = knn.score(test_modules, test_target)
        print(params)
        print(acc)

        if acc > best_acc:
            best_acc = acc
            best_params = params

    # save results to file
    with open(f'all_modules_results/{model}/accuracy_score', 'w') as f:
        f.write(str(best_acc)+'\n')
    with open(f'all_modules_results/{model}/best_params', 'w') as f:
        f.write(str(best_params)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str
    )

    flags = parser.parse_args()
    all_modules_run(flags.model)
