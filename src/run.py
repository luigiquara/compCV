from torchvision.transforms import v2 as transforms
from torchvision import models

from avalanche.benchmarks.datasets import CUB200

from save_activations import SaveActivations


def run_resnet():

    # load and preprocess CUB dataset
    preprocessing = transforms.Compose([
    transforms.ToImageTensor(),
    transforms.ConvertImageDtype(),
    # mean and std values from https://pytorch.org/hub/pytorch_vision_resnet/
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    cub = Cub200(root='/home/lquarantiello/compCV', transform=preprocessing, download=False)


if __name__ == '__main__':
    run_resnet()