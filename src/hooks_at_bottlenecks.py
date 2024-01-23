import argparse
from tqdm import tqdm
import wandb
import torch
from torch import nn

from utils import get_model, load_cub
from torchinfo import summary

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# select a given number of single bottlenecks
# add an avgpool on top, as the original network
class PartialResNet(nn.Module):
    def __init__(self, model, n_rm_bottlenecks):
        super(PartialResNet, self).__init__()

        backbone = get_model(model)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # remove last two elements, i.e. avgpool and linear classifier
        layers = list(backbone.children())[:-2]

        # each block is composed by several bottlenecks
        for index, block in enumerate(reversed(layers)):

            # if the number of bottlenecks is bigger than the ones present in the current block
            # go to the next block
            if n_rm_bottlenecks >= len(block):
                n_rm_bottlenecks -= len(block)
                del layers[-1]
                continue
            
            # else, if the number of bottlenecks to remove is contained in the current block
            # we got the index of the block in layers list, and the index of the last bottleneck
            # hence, we can go out of the loop
            break
            
        layers[-1] = layers[-1][:len(layers[-1]) - n_rm_bottlenecks]
        self.partial_net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.avgpool(self.partial_net(x))
        return out

def hooks_at_bottlenecks(model):
    device = torch.device('cuda')
    tr_loader, test_loader = load_cub(batch_size=32)
   
    # remove single bottlenecks modules
    for n_rm_bottlenecks in range(10):
        resnet = PartialResNet(model, n_rm_bottlenecks).to(device)
        print(f'Removing {n_rm_bottlenecks} single bottlenecks')

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
        print(f'Accuracy when removing {n_rm_bottlenecks} single bottlenecks: {acc}')


        wandb.config.update({'model': model,
                             'batch_size': 32,
                             'n_neighbors': 20,
                             'weights_knn': 'distance'})
        wandb.log({'accuracy': acc, 'bottlenecks removed': n_rm_bottlenecks})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str
    )
    flags = parser.parse_args()
    model = flags.model

    wandb.init(project = 'CV & Compositionality',
               group='hooks_at_bottlenecks')
    hooks_at_bottlenecks(model)
