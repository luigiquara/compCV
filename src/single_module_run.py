from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset
from avalanche.benchmarks.datasets import CUB200

from torchvision.transforms import v2 as transforms
from torchvision import models
from sklearn.neighbors import KNeighborsClassifier

from save_activations import SaveFeatures

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# Define the upperbound in our setting
# Use the entire model - i.e. compose a single module - to see the performance on CUB
def single_module_run():

    # load and preprocess CUB dataset
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

    #cub_training = Subset(cub_training, torch.arange(0,20))
    #cub_test = Subset(cub_test, torch.arange(0,20))

    tr_loader = DataLoader(cub_training, batch_size=len(cub_training))
    test_loader = DataLoader(cub_test, batch_size=len(cub_test))

    # load the model & register the hook at the last layer before classification
    resnet = models.resnet152(weights='ResNet152_Weights.DEFAULT', progress=True)
    resnet.eval()
    extractor = SaveFeatures(resnet, ['avgpool'])

    with torch.no_grad():
        for batch, training_target in tqdm(tr_loader):
            features = extractor(batch)
        training_features = torch.squeeze(features['avgpool'])

        for batch, test_target in tqdm(test_loader):
            features = extractor(batch)
        test_features = torch.squeeze(features['avgpool'])



    # grid search on the knn
    # get the configuration with the best accuracy 
    params = {
        'n_neighbors': [5, 10, 20, 30, 40, 50],
        'weights': ['uniform', 'distance']
    } 
    grid = GridSearchCV(KNeighborsClassifier(), params, scoring='accuracy')
    grid.fit(training_features.numpy(), training_target.numpy())

    print(grid.best_params_)
    predictions = grid.predict(test_features)
    acc = accuracy_score(predictions, test_target)
    print(acc)

    import code; code.interact(local=locals())




if __name__ == '__main__':
    single_module_run()
