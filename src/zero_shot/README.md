# Testing Zero-Shot Capabilities

In the context of image classification tasks, we want to test the zero-shot capabilities of different pretrained models on a set of downstream tasks.
At the moment, the architecture used is the ResNet50, pretrained on ImageNet, COCO, CIFAR10 and CIFAR100.
Currently, these models are tested on StanfordCars, Oxford102Flowers and INaturalist2021.

## Noted on the Models
### ResNet50 - COCO
We were not able to find online a ResNet50 architecture trained on COCO dataset.
We decided to take the FasterRCNN model, pretrained on COCO, extract the ResNet backbone and use it for the successive experiments.<br/>
The instructions we followed to extract the ResNet from the FasterRCNN model are available at: https://discuss.pytorch.org/t/feature-extracting-from-resnet-pretrained-on-coco/82010/3

## Notes on the Datasets
### StanfordCars
StanfordCars is currently not availble at its homepage: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
We downloaded it from Kaggle, following the procedure explained in https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616. Thank you @thefirebanks for providing those helpful instructions.

### INaturalist 2021
The INaturalist 2021 (https://github.com/visipedia/inat_comp/tree/master/2021) contains about 2.7M images of animals, plants, ecc.
The task in our case was to predict the *full* target, *i.e.* the species.

Since the dataset is too big (and it would take hours to perform a forward pass), we decided to use a subset of the images, keeping the ones tagged as *Amphibia*.
Therefore, we ended up with 170 classes, 46252 train images and 1700 test images.

## Classification Head
To perform a different task *wrt* the pretraining one, we needed to substitute the classification head.
We used a LogisticRegression model, to keep it simple and to test directly the features of the model, without any additional representation power.
For our experiments, we fixed *max_iter* = 400, that we found to be a good trade-off between accuracy and time complexity.

## Results
### Single-Model
Here, the results obtained.
On the rows, we put the different models we tested, on the columns the downstream tasks.
The results displayed here are the values of the accuracy on the test set of the respective tasks.

|                     | StanfordCars | Oxford102Flowers | INat21 |
|---------------------|--------------|------------------|--------|
| ResNet50 - ImageNet | 54%          | 82%              | 35%    |
| ResNet50 - COCO     | 40%          | 71%              | 20%    |
| ResNet50 - CIFAR100 | 9%           | 44%              | 7%     |
| ResNet50 - CIFAR10  | 5%           | 19%              | 5%     |

## Take-Home Message
Models pretrained on CIFAR10/100 are not good enough - 32x32 images are too small for the training.
The Flowers dataset seems a little simpler - probably in ImageNet there are several images of flowers.

**TODO**:
+ Check perfomance of fine-tuned models on downstream tasks
+ Use another architecture, *e.g.* ViT
+ Split ImageNet in half, train a model on each half and use these two as *backbones*.
+ Use two backbones, *e.g.* ResNet-ImageNet & ResNet-COCO, with a single classifier for each downstream task.
