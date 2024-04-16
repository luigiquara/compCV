'''Train experts on cropped GQA

Cropped GQA contains 21 classes.
Train expert models on them, to use them later in CGQA
'''

import argparse
import numpy as np
import random
import time
import wandb

from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18

from training import Trainer

# the classes from CR_GQA, derived following the instruction from CGQA/continual json files
classes = ['plate', 'shirt', 'building', 'sign', 'grass', 'car', 'table', 'chair', 'jacket', 'shoe', 'flower', 'pants', 'helmet', 'bench', 'pole', 'leaves', 'wall', 'door', 'fence', 'hat', 'shorts']

# default transformations
# ImageNet normalization
transform = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'eval': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

def _get_subset_idx(dataset, current_classes_idxs):
    '''Get indexes to create subsets for (1 vs all)-classification

    Get all samples  for current classes, and a random sample of other classes indexes, of the same length.

    Parameters
    ----------
    dataset:
        The dataset to subset
    current_classes_idxs: List[Int]
        The list of current classes to learn, in the form of indexes
    
    Return
    ------
    List[Int]:
        A list of indexes, from which extract the subset
    '''

    cc_idx = [i for i in range(len(dataset)) if dataset.targets[i] in current_classes_idxs]
    remaining_idx = list(set(range(len(dataset))) - set(cc_idx))
    no_cc_idx = random.choices(remaining_idx, k=len(cc_idx))
    return cc_idx + no_cc_idx



def run(param):
    # load dataset
    e_tr_set = ImageFolder(param.dataset_path+'train/', transform=transform['train'])
    e_val_set = ImageFolder(param.dataset_path+'val/', transform=transform['eval'])
    e_test_set = ImageFolder(param.dataset_path+'test/', transform=transform['eval'])

    divided_classes = np.array_split(np.array(classes), param.n_experts)

    # we need a number of training processes equal to the number of experts
    for i in range(param.n_experts):
        if param.log:
            run = wandb.init(
                  project='CR_GQA',
                  group=f'Expert {i+1}/{param.n_experts}',
                  config = {
                      'max_epochs': param.max_epochs,
                      'batch_size': param.batch_size,
                      'lr': param.lr
                  })
        
        # the classes that the current expert has to learn
        current_classes = list(divided_classes[i])
        current_classes_idxs = [e_tr_set.class_to_idx[c] for c in current_classes]
        print(f'Training Expert {i+1}/{param.n_experts} on {current_classes}')

        # create a subset of the entire dataset
        # select all the samples for current classes, and a percentage of the other classes
        tr_idxs = _get_subset_idx(e_tr_set, current_classes_idxs) 
        val_idsx = _get_subset_idx(e_val_set, current_classes_idxs)
        test_idxs = _get_subset_idx(e_test_set, current_classes_idxs)
        tr_set = Subset(e_tr_set, tr_idxs)
        val_set = Subset(e_val_set, val_idsx)
        test_set = Subset(e_test_set, test_idxs)

        # create dataloaders
        tr_loader = DataLoader(tr_set, batch_size=param.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=param.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=param.batch_size, shuffle=False)

        # create the model
        # change the classifier, to perform (1 vs All)-classification
        # we want to classify the current classes, plus the 'other' class
        model = resnet18()
        fc_in_features = model.fc.in_features
        model.fc = nn.Linear(fc_in_features, len(current_classes)+1)

        # training process
        # inside, the labels are preprocessed for (1 vs all)-classification
        save_path = param.save_root + f'expert_{i+1}/{run.name}'
        optimizer = Adam(model.parameters(), lr=param.lr)
        trainer = Trainer(model, nn.CrossEntropyLoss(), optimizer, current_classes=current_classes_idxs, save_path=save_path, device='cuda', log=param.log)
        results = trainer.train(tr_loader, val_loader, num_classes=len(current_classes)+1, epochs=param.max_epochs)

        run.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_experts', type=int, default=7)
    parser.add_argument('--dataset_path', type=str, default='/disk4/lquarantiello/CR_GQA/')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_root', type=str, default='/disk4/lquarantiello/compCV/CFST-compCV/experts/')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--no_log', action='store_false', dest='log')

    param = parser.parse_args()
    print(param)

    run(param)
