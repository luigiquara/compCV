from tqdm import tqdm

from avalanche.benchmarks.datasets import CUB200
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.transforms import v2 as transforms

import torch
from torchvision import models

from save_features import SaveFeatures


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

# apply the pooling to all modules head
# both max and avg pooling
# then concat all the pooled activations
def pool_and_concat(features):
    # pooling of training features
    # if kernel_size == matrix_size, it returns a single value
    avg_pooled = []
    max_pooled = []
    for key, feature in features.items():
        assert feature.shape[2] == feature.shape[3], feature.shape
        avg_layer = torch.nn.AvgPool2d(kernel_size = feature.shape[3])
        max_layer = torch.nn.MaxPool2d(kernel_size = feature.shape[3])

        # needed to change the view to apply pad_sequence
        # it needs the last dimension to be equal
        avg_pooled.append(torch.squeeze(avg_layer(feature)).view((-1, feature.shape[0])))
        max_pooled.append(torch.squeeze(max_layer(feature)).view(-1, feature.shape[0]))

    # concat the activations from all layers
    avg_pooled = torch.cat(avg_pooled).view((feature.shape[0], -1))
    max_pooled = torch.cat(max_pooled).view(feature.shape[0], -1)
    return avg_pooled, max_pooled


# take a resnet model
# forward pass on cub
# return the features for training and validation sets
def get_modules_features(model):
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
    cub_training, cub_validation = random_split(cub_training, [0.7, 0.3])
    #cub_test = CUB200(root='/home/lquarantiello/compCV', transform=preprocessing, train=False, download=False)

    #cub_training = Subset(cub_training, torch.arange(0,30))
    #cub_validation = Subset(cub_validation, torch.arange(0,30))
    #cub_test = Subset(cub_test, torch.arange(0,20))

    tr_loader = DataLoader(cub_training, batch_size=len(cub_training))
    val_loader = DataLoader(cub_validation, batch_size=len(cub_validation))
    #test_loader = DataLoader(cub_test, batch_size=len(cub_test))

    print(f'\n\nUsing {model}\n\n')
    resnet = get_model(model)
    
    resnet.eval()
    extractor = SaveFeatures(resnet, layers=get_relu_layers(resnet))

    with torch.no_grad():
        for batch, training_target in tqdm(tr_loader):
            training_features = extractor(batch)

        for batch, val_target in tqdm(val_loader):
            val_features = extractor(batch)

    avg_pooled_tr_features, max_pooled_tr_features = pool_and_concat(training_features)
    avg_pooled_val_features, max_pooled_val_features = pool_and_concat(val_features)

    tr_features = {'avg_pool': avg_pooled_tr_features, 'max_pool': max_pooled_tr_features}
    val_features = {'avg_pool': avg_pooled_val_features, 'max_pool': max_pooled_val_features}
    return tr_features, val_features, training_target, val_target