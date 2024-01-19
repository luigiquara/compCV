import argparse
from tqdm import tqdm
import wandb
import torch
from torch import nn

from utils import get_model, load_cub
from save_features import SaveFeatures

class Resnet4CUB(nn.Module):
    def __init__(self, model, n_classes):
        super(Resnet4CUB, self).__init__()

        self.backbone = get_model(model)
        # change the classification head
        self.linear_head = nn.Linear(self.backbone.fc.in_features, n_classes)
        self.backbone.fc = self.linear_head

    def forward(self, x):
        logits = self.backbone(x)
        return logits

def resnet_mlp(model):
    device = torch.device('cuda')
    tr_loader, test_loader = load_cub(batch_size=8)
    wandb.config.update({'model': model, 'batch_size': 8})

    resnet = Resnet4CUB(model, n_classes=200)
    resnet.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)

    # training loop
    resnet.train()
    epoch_loss = float('inf')
    previous_loss = float('inf')
    epochs = 0
    delta = 10
    first = True

    # stop when you are getting no more improvements
    while first or abs(epoch_loss - previous_loss) > delta:
        first = False
        previous_loss = epoch_loss
        epoch_loss = 0.0

        for i, data in enumerate(tqdm(tr_loader)):
            mb, target = data
            mb = mb.to(device)
            target = target.to(device)

            logits = resnet(mb)
            loss = loss_fn(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        print(f'Epoch: {epochs}')
        print(f'Current loss: {epoch_loss}')
        wandb.log({'training_loss': epoch_loss})
        epochs += 1

    wandb.log({'number of epochs': epochs})

    # test
    resnet.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        total_loss = 0
        for mb, target in tqdm(test_loader):
            # compute test accuracy
            mb = mb.to(device)
            target = target.to(device)

            logits = resnet(mb)
            loss = loss_fn(logits, target)
            total_loss += loss.item()

            _, preds = torch.max(logits, dim=1)

            n_correct += (preds == target).sum()
            n_samples += preds.size(0)
    
    wandb.log({'test_loss': total_loss, 'accuracy': n_correct/n_samples})        
    print(f'n_correct: {n_correct}')
    print(f'n_samples: {n_samples}')
    print(f'Accuracy: {n_correct/n_samples}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str
    )
    flags = parser.parse_args()
    model = flags.model

    wandb.init(project = 'CV & Compositionality', group='baseline_finetuned')
    resnet_mlp(model)