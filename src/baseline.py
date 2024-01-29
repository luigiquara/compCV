import argparse
from tqdm import tqdm
import wandb
import torch

from utils import get_model, load_cub
from save_features import Extractor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def zero_shot(model, classifier):
    device = torch.device('cuda')
    resnet = get_model(model).to(device)
    tr_loader, test_loader = load_cub(batch_size=32)

    resnet.eval()
    # extract features from the last layer before the classification head
    extractor = Extractor(resnet, layers_id=['avgpool'])

    #training_features = {'avgpool': torch.tensor([]).to(device)}
    training_features = {'avgpool': torch.tensor([])}
    training_target = []
    #test_features = {'avgpool': torch.tensor([]).to(device)}
    test_features = {'avgpool': torch.tensor([])}
    test_target = []


    # forward pass
    with torch.no_grad():
        # pass on training set
        for mb, target in tqdm(tr_loader):
            mb = mb.to(device)
            features = extractor(mb)
            training_features['avgpool'] = torch.cat((training_features['avgpool'], features['avgpool']), 0)
            training_target.append(target)
            
        # pass on test set
        for mb, target in tqdm(test_loader):
            mb = mb.to(device)
            features = extractor(mb)
            test_features['avgpool'] = torch.cat((test_features['avgpool'], features['avgpool']), 0)
            test_target.append(target)
        
        training_features = training_features['avgpool'].squeeze()
        training_target = torch.cat(training_target)
        test_features = test_features['avgpool'].squeeze()
        test_target = torch.cat(test_target)

    if classifier == 'knn':
        print('Using KNeighborsClassifier')
        classifier = KNeighborsClassifier(n_neighbors=20, weights='distance')
        wandb.config.update({
            'batch_size': 32,
            'classifier': 'knn',
            'n_neighbors': 20,
            'weights_knn': 'distance'
        })
    elif classifier == 'logreg':
        print('Using LogisticRegression')
        classifier = LogisticRegression(max_iter=500)
        wandb.config.update({
            'batch_size': 32,
            'classifier': 'LogisticRegression',
            'max_iter': 500
        })
    
    # fit the selected classifier and get the test accuracy
    classifier.fit(training_features.numpy(force=True), training_target.numpy(force=True))
    preds = classifier.predict(test_features.numpy(force=True))
    acc = accuracy_score(preds, test_target.numpy(force=True))
    print(acc)

    wandb.log({'accuracy': acc})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str
    )
    parser.add_argument(
        '--classifier',
        type=str
    )
    flags = parser.parse_args()
    model = flags.model
    classifier = flags.classifier

    wandb.init(project = 'CV & Compositionality')
    zero_shot(model, classifier)
