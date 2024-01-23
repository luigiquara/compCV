import argparse
from tqdm import tqdm
import wandb
import torch
from torch import nn

from utils import get_model, load_cub
from torchinfo import summary

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# select a given number of bottleneck blocks
# add an avgpool on top, as the original network
class PartialResNet(nn.Module):
    def __init__(self, model, n_blocks):
        super(PartialResNet, self).__init__()

        backbone = get_model(model)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # get the partial network
        # first 4 layers are common, then bottlenecks start
        # "blocks" refers to bottleneck blocks
        layers = list(backbone.children())[:4+n_blocks]
        self.partial_net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.avgpool(self.partial_net(x))
        return out

def hooks_at_blocks(model):
    device = torch.device('cuda')
    tr_loader, test_loader = load_cub(batch_size=32)
   
    # using 4, 3, 2, 1 and 0 "bottleneck" blocks
    # 4 blocks is the entire network
    for n_blocks in range(4, 0, -1):
        resnet = PartialResNet(model, n_blocks).to(device)
        print(f'Using {n_blocks} bottleneck blocks')

        training_features = torch.tensor([]).to(device)
        training_target = []
        test_features = torch.tensor([]).to(device)
        test_target = []
        resnet.eval()
        with torch.no_grad():
            # pass on training set
            print('Starting training')
            for mb, target in tqdm(tr_loader):
                mb = mb.to(device)
                features = resnet(mb)
                training_features = torch.cat((training_features, features), 0)
                training_target.append(target)

            print('Starting testing')
            # pass on test set
            for mb, target in tqdm(test_loader):
                mb = mb.to(device)
                features = resnet(mb)
                test_features = torch.cat((test_features, features), 0)
                test_target.append(target)

            training_features = training_features.squeeze()
            training_target = torch.cat(training_target)
            test_features = test_features.squeeze()
            test_target = torch.cat(test_target)

        # knn fitting
        print('Fitting the knn')
        knn = KNeighborsClassifier(n_neighbors=20, weights='distance')
        knn.fit(training_features.numpy(force=True), training_target.numpy(force=True))
        preds = knn.predict(test_features.numpy(force=True))
        acc = accuracy_score(preds, test_target.numpy(force=True))
        print(f'Accuracy for {n_blocks} bottlenecks: {acc}')


        wandb.config.update({'model': model,
                             'batch_size': 32,
                             'n_neighbors': 20,
                             'weights_knn': 'distance'})
        wandb.log({'accuracy': acc, 'blocks_used': n_blocks})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str
    )
    flags = parser.parse_args()
    model = flags.model

    wandb.init(project = 'CV & Compositionality', group='hooks_at_blocks')
    hooks_at_blocks(model)
