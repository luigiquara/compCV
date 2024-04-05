import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

import torch
from torch import nn
from torch.optim import Adam, SGD
from torchvision import transforms

import ds.cgqa_general as cgqa
from train_models import LResNet


class CompResNet(L.LightningModule):
    def __init__(self, resnets, cl_cfg):
        super().__init__()

        # the input to the classifier of a single resnet
        # we assume to have models with the same architecture
        single_num_filters = resnets[0].fc.in_features

        self.backbones = [] # list of models to use
        for resnet in resnets: _remove_classifier(resnet)

        self.cl_cfg
        self.classifier = _init_classifier()

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer_cfg = optimizer_cfg

    def forward(self, x):
        with torch.no_grad():
            features = []
            for bb in self.backbones: features.append(bb(x).flatten(1))
            features = torch.cat(features)
    def training_step(self):
        pass

    def validation_step(self):
        pass

    def test_step(self):
        pass

    def _remove_classifier(resnet):
        layers = list(resnet.children())[:-1]
        feature_extractor = nn.Sequential(*layers)
        feature_extractor.eval()

        self.backbones.append(feature_extractor)

    def _init_classifier():
        if self.cl_cfg.name == 'knn':
            classifier = KNeighborsClassifier(n_neighbors=cl_cfg.n_neighbors, weights='distance')
        elif self.cl_cfg.name == 'logreg':
            classifier = LogisticRegression(max_iter=cl_cfg.max_iter)

        return classifier

def _init_classifier(cfg):
    if cfg.name == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=cfg.n_neighbors, weights='distance')
    elif cfg.name == 'logreg':
        classifier = LogisticRegression(max_iter=cfg.max_iter)
    
    return classifier

def forward_pass(models, loader):
    for m in models:
        m.to('cuda')
        m.eval()

    features = []
    for idx, x, y in tqdm(enumerate(loader)):
        x = x.to('cuda')

        f = []
        for m in models:
            f.append(m(x))
        features[idx] = torch.cat(f)

    return features



@hydra.main(config_path='config', config_name='config', version_base=None)
def run(cfg: DictConfig):

    # preprocessing transformations
    # from CFST appendix
    tr_preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for mode in ['sys', 'pro', 'sub', 'non', 'noc']:
        print(f'mode: {mode}')

        benchmark = cgqa.fewshot_testing_benchmark(n_experiences=cfg.n_experiences, seed=42, dataset_root='/disk3/lquara/', mode=mode, train_transform=tr_preprocess, eval_transform=test_preprocess)

        for exp_id in range(cfg.n_experiences):

            # load models & classifier
            backbones = load_models(strategy = 'all') # the list of pretrained models to use for the current experience
            classifier = _init_classifier(cfg.classifier)

            # get the data
            train_loader = DataLoader(benchmark.train_datasets[exp_id], batch_size=cfg.train.batch_size, shuffle=True, num_workers=11)
            val_loader = DataLoader(benchmark.val_datasets[exp_id], batch_size=cfg.train.batch_size, num_workers=11)
            test_loader = DataLoader(benchmark.test_datasets[exp_id], batch_size=cfg.train.batch_size, num_workers=11)

            # training of the classifier
            forward_pass(backbones, train_loader)