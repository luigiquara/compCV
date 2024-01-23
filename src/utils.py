import torch
from torchvision import models
from avalanche.benchmarks.datasets import CUB200
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2 as transforms

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

# load and preprocess CUB dataset
def load_cub(batch_size):
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

    #cub_training = Subset(cub_training, torch.arange(0,60))
    #cub_test = Subset(cub_test, torch.arange(0,40))

    tr_loader = DataLoader(cub_training, batch_size=batch_size)
    test_loader = DataLoader(cub_test, batch_size=batch_size)

    return tr_loader, test_loader
