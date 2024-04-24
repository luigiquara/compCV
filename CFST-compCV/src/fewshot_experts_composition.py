'''Take the k top experts for the dataset and compose them with a fewshot trained classifier
'''

import argparse
import numpy as np
from typing import List
import wandb

import torch
from torch import nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from ds import cgqa_general as cgqa
from training import Trainer
from zero_shot_experts import _load_experts

# the model that composes different backbones
# with a new mlp classifier to train 
class FrankensteinMonster(nn.Module):
    def __init__(self, backbones: List[nn.Module], n_classes: int):
        super(FrankensteinMonster, self).__init__()

        # remove old classification heads
        # and freeze backbones
        for i, b in enumerate(backbones): backbones[i] = self._remove_classifier_and_freeze(b)
        self.backbones = backbones

        # define the new classifier
        in_features = 512*len(self.backbones)
        self.classifier = nn.Linear(in_features, n_classes)

    def forward(self, x):
        device = 'cpu' if x.get_device == -1 else 'cuda'

        # concat the output of the backbones
        out = []
        for b in self.backbones:
            out.append(b(x).tolist())
        
        # (batch_size, len(backbones)*512, 1, 1)
        out = torch.cat(list(torch.Tensor(o) for o in out), dim=1).to(torch.device(device)).squeeze()
        out = self.classifier(out)

        return out

    def _remove_classifier_and_freeze(self, model):
        new_model = nn.Sequential(*list(model.children())[:-1])
        for param in new_model.parameters(): param.requires_grad = False
        return new_model

def select_top_experts(loader: DataLoader, experts: List[nn.Module], n_experts: int, device: str) -> List[int]:
    '''Given a dataset and the list of experts, select the n_experts which are mostly excited.

    For each expert, take the sum of the max logits for each sample.
    Then, selected the experts with the top values.

    Return
    ------
    List[int]
        The idxs of the top experts.
    '''

    expert_excitement = np.zeros(len(experts))
    for X, y in loader:
        X = X.to(torch.device(device))
        y = y.to(torch.device(device))

        for i, e in enumerate(experts):
            # forward pass on the mb
            logits = softmax(e(X), dim=1)
            confidence, pred = torch.max(logits, dim=1)

            # get the sum of the max confidence over the mb
            # update the excitement of the selected expert
            expert_excitement[i] += torch.sum(confidence).item()
    
    # sort the experts in descening order and get the top ones
    return  np.argsort(expert_excitement)[::-1][:n_experts]

def run(params):
    if params.log:
        run = wandb.init(
            project = 'Expert Composition',
            config = {
                'max_epochs': params.max_epochs,
                'lr': params.lr,
                'n_experts': params.n_experts,
                'n_classes': params.n_classes
            }
        )

    # get one experience with n_classes
    # 10/5/10 samples per classes for train/val/test set
    benchmark = cgqa.fewshot_testing_benchmark(n_experiences=int(100/params.n_classes), n_way=params.n_classes, mode='sys', image_size=(196,196), task_offset=1, dataset_root='/disk3/lquara')

    # create dataloaders
    train_loader = DataLoader(benchmark.train_datasets[0], batch_size=params.batch_size)
    val_loader = DataLoader(benchmark.val_datasets[0], batch_size=params.batch_size)
    test_loader = DataLoader(benchmark.test_datasets[0], batch_size=params.batch_size)

    # load the experts
    model_names = ['pleasant-smoke-27', 'resilient-valley-28', 'fanciful-cosmos-29', 'ethereal-firefly-30', 'hearty-feather-31', 'twilight-rain-32', 'northern-meadow-33']
    filepaths = [params.expert_path + f'expert_{i+1}/' + name for i, name in enumerate(model_names)]
    experts = _load_experts(filepaths, classes_per_experts=params.classes_per_expert, device=params.device)

    # select top-k experts given the train_set
    # and remove their classifiers
    selected_experts_idxs = select_top_experts(train_loader, experts, params.n_experts, params.device)
    print(f'Using {len(selected_experts_idxs)} experts')

    # create the compositional model and train it
    if params.log: save_path = params.save_root + run.name
    else: save_path = params.save_root + 'aaaa'
    exp_comp = FrankensteinMonster([experts[i] for i in selected_experts_idxs], n_classes=params.n_classes)
    print(f'Number of trainable parameters: {sum(p.numel() for p in exp_comp.parameters() if p.requires_grad)}')

    optimizer = torch.optim.Adam(exp_comp.parameters(), lr=params.lr)

    trainer = Trainer(exp_comp, nn.CrossEntropyLoss(), optimizer, save_path=save_path, device=params.device, log=params.log)
    results = trainer.train(train_loader, val_loader, num_classes=params.n_classes, epochs=params.max_epochs)
    test_results = trainer.test(test_loader, num_classes=params.n_classes)

    run.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_experts', type=int, default=3, help='The number of expert to select for the composition')
    parser.add_argument('--expert_path', type=str, default='/disk4/lquarantiello/compCV/CFST-compCV/experts/')
    parser.add_argument('--classes_per_expert', type=int, default=4)
    parser.add_argument('--n_classes', type=int, default=100)

    # generic params
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--no_log', action='store_false', dest='log')
    parser.add_argument('--device', type=str, default='cuda')

    # trainer params
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_root', type=str, default='/disk4/lquarantiello/compCV/CFST-compCV/experts_composition/')

    params = parser.parse_args()
    print(params)

    run(params)