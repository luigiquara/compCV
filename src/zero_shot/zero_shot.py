"""
Test zero-shot capabilities of pretrained models.
Currently supporting:
+ Models: ResNet
+ Pretrained on: ImageNet, CIFAR10, CIFAR100
+ Downstream task: StanfordCars, INaturalist, Oxford102Flowers
"""

import argparse
import copy
import numpy as np
from tqdm import tqdm

from datasets import load_dataset
import detectors # needed together with timm to load cifar models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import Flowers102, INaturalist, ImageFolder, StanfordCars
from torchvision.models import resnet50
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from transformers import ResNetModel 
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndNoAttention

def _from_fasterrcnn_to_resnet():
    '''Extract the ResNet backbone from FasterRCNN

    Online, there is plain ResNet50 model pretrained on COCO dataset.
    There is a FasterRCNN model trained on COCO, which has a ResNet backbone and some additional layers.
    This method is to extract the ResNet backbone and use it for classification.
    Taken from https://discuss.pytorch.org/t/feature-extracting-from-resnet-pretrained-on-coco/82010/3.

    Returns
    -------
    model : torchvision.models.resnet.ResNet
        The ResNet backbone extracted from FasterRCNN model trained on COCO
    '''

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    resnet = model.backbone.body

    # Check for all FrozenBN layers
    bn_to_replace = []
    for name, module in resnet.named_modules():
        if isinstance(module, torchvision.ops.misc.FrozenBatchNorm2d):
            #print('adding ', name)
            bn_to_replace.append(name)

    # Iterate all layers to change
    for layer_name in bn_to_replace:
        # Check if name is nested
        *parent, child = layer_name.split('.')
        # Nested
        if len(parent) > 0:
            # Get parent modules
            m = resnet.__getattr__(parent[0])
            for p in parent[1:]:    
                m = m.__getattr__(p)
            # Get the FrozenBN layer
            orig_layer = m.__getattr__(child)
        else:
            m = resnet.__getattr__(child)
            orig_layer = copy.deepcopy(m) # deepcopy, otherwise you'll get an infinite recursion
        # Add your layer here
        in_channels = orig_layer.weight.shape[0]
        bn = nn.BatchNorm2d(in_channels)
        with torch.no_grad():
            bn.weight = nn.Parameter(orig_layer.weight)
            bn.bias = nn.Parameter(orig_layer.bias)
            bn.running_mean = orig_layer.running_mean
            bn.running_var = orig_layer.running_var
        m.__setattr__(child, bn)

    # Fix the bn1 module to load the state_dict
    resnet.bn1 = resnet.bn1.bn1

    # Create reference model and load state_dict
    reference = resnet50()
    reference.load_state_dict(resnet.state_dict(), strict=False)

    return reference

def get_model(model_name, pretraining_set):
    '''Get a pretrained model, given its name and the wanted pretraining dataset

    Parameters
    ----------
    model_name : str
        The name of the model. Currently supporting resnet50
    pretraining_set : str
        The name of the pretraining dataset. Currently supporting imagenet, cifar10, cifar100
    
    Returns
    -------
    model : torch.nn.Sequential
        The pretrained PyTorch model, together with the weights corresponding to the pretraining dataset.
        We download the models using timm, only removing the last fully connected layer
    prepocess : transforms.Compose
        The set of torchvision tranformations, based on the selected model
    '''

    if pretraining_set == 'imagenet':
        complete_model = timm.create_model('resnet50', pretrained=True)

        # preprocessing transformations for the imagenet model
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    elif pretraining_set == 'cifar10':
        complete_model = timm.create_model('resnet50_cifar10', pretrained=True)

        # from timm model.pretraining_cfg
        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
        ])

    elif pretraining_set == 'cifar100':
        complete_model = timm.create_model('resnet50_cifar100', pretrained=True)

        # from timm model.pretraining_cfg
        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])

    elif pretraining_set == 'coco':
        complete_model = _from_fasterrcnn_to_resnet()

        preprocess = transforms.Compose([
            transforms.Resize((640, 480)), #median image size in COCO
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # TODO: raise exception
        pass

    model = torch.nn.Sequential(*list(complete_model.children())[:-1]) # remove classification head

    return model, preprocess

def get_dataset(task_dataset, preprocess):
    '''Return the dataloaders for a given downstream task dataset.

    Parameters
    ----------
    task_dataset : str
        The name of the dataset for the downstream task. Currently supporting INaturalist, Oxford102Flowers
    preprocess : torchvision.transforms.Compose
        The set of torchvision transformations, needed for the selected model.

    Returns
    -------
    tr_loader : torch.utils.data.DataLoader
        DataLoader for the training set of the downstream task.

    test_loader : torch.utils.data.DataLoader
        DataLoader for the test set of the downstream task.
    '''

    if task_dataset == 'stanford_cars':
        tr_loader = DataLoader(StanfordCars('/disk3/lquara/', transform=preprocess, split='train'), batch_size=32)
        test_loader = DataLoader(StanfordCars('/disk3/lquara/', transform=preprocess, split='test'), batch_size=32)
    elif task_dataset == 'inaturalist':
        #tr_loader = DataLoader(INaturalist(root='/disk3/lquara/INaturalist', version='2021_train_amph', target_type='full', transform=preprocess, download=False), batch_size=128)
        #test_loader = DataLoader(INaturalist(root='/disk3/lquara/INaturalist', version='2021_valid', transform=preprocess, download=False), batch_size=128)
        tr_loader = DataLoader(ImageFolder('/disk3/lquara/INaturalist/2021_train_amph', transform=preprocess), batch_size=32)
        test_loader = DataLoader(ImageFolder('/disk3/lquara/INaturalist/2021_valid_amph', transform=preprocess), batch_size=32)
    elif task_dataset == 'oxford102flowers':
        tr_loader = DataLoader(Flowers102(root='/disk3/lquara', split='train', transform=preprocess, download=True), batch_size=32)
        test_loader = DataLoader(Flowers102(root='/disk3/lquara', split='test', transform=preprocess, download=True), batch_size=32)
    else:
        # TODO: raise exception
        pass

    return tr_loader, test_loader

def get_classifier(head):
    '''Define the classification head.

    Parameters
    ----------
    head : str
        The name of the classification head. Currently supporting LogisticRegression and KNeighborsClassifier

    Returns
    -------
    sklearn-classifier
        The classification head
    '''

    if head == 'logistic_regression':
        classifier = LogisticRegression(max_iter=400)
    elif head == 'knn':
        classifier = KNeighborsClassifier()
    else:
        #TODO: raise an exception
        raise NotImplemented

    return classifier

def forward_pass(model, dataloader):
    '''Perform one pass on a dataset, returning the features.

    Parameters
    ----------
    model : torch.nn.Sequential
        The model to use, already instantiated.
    dataloader : torch.utils.data.DataLoader
        The DataLoader to use

    Returns
    -------
    tr_features : torch.Tensor
        The output of the model on the entire DataLoader
    tr_target : torch.Tensor
        The labels of each entry of the DataLoader
    '''

    features, target = [], []

    model.eval()
    model.cuda()
    with torch.no_grad():
        for mb, t in tqdm(dataloader, desc='forward pass'):
            mb = mb.cuda()
            t = t.cuda()

            out = model(mb).squeeze()
            
            # depending on the model in use
            if isinstance(out, torch.Tensor):
                features.append(out.numpy(force=True))
            elif isinstance(out, BaseModelOutputWithPoolingAndNoAttention):
                features.append(out.pooler_output.squeeze().cpu().detach().numpy())
            target.append(t.cpu().detach().numpy())
    
    # convert everything to PyTorch tensors
    features = np.concatenate(features)
    target = np.concatenate(target)

    return features, target

def main(model_name, pretraining_set, task_dataset, head, log_file):
    model, processor = get_model(model_name, pretraining_set)
    tr_loader, test_loader = get_dataset(task_dataset, processor)
    classifier = get_classifier(head)

    # classification head training
    tr_features, tr_target = forward_pass(model, tr_loader)
    print('Fitting of the classifier')
    classifier.fit(tr_features, tr_target)

    # evaluation
    test_features, test_target = forward_pass(model, test_loader)
    acc = classifier.score(test_features, test_target)

    print(acc)

    # logging
    with open(log_file, 'a') as fp:
        fp.write(f'{model_name} pretrained on {pretraining_set}\nUsing {head}\nAccuracy on {task_dataset}: {acc}\n\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        type=str
    )
    parser.add_argument(
        '--pretraining_set',
        type=str,
    )
    parser.add_argument(
        '--task_dataset',
        type=str
    )
    parser.add_argument(
        '--head',
        type=str,
        help='The type of classification head to use'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        help='The filepath where to save results'
    )

    args = parser.parse_args()
    main(**vars(args))