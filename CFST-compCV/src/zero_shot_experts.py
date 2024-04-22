'''Load expert models trained on CR_GQA, and use them zero-shot on CGQA test set
'''

import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from typing import List, Union

import torch
from torch import nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Subset
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_auroc, multiclass_confusion_matrix
from torchvision import transforms
from torchvision.models import resnet18

from ds import cgqa_general as cgqa

# the classes from CR_GQA, derived following the instruction from CGQA/continual json files
classes = ['plate', 'shirt', 'building', 'sign', 'grass', 'car', 'table', 'chair', 'jacket', 'shoe', 'flower', 'pants', 'helmet', 'bench', 'pole', 'leaves', 'wall', 'door', 'fence', 'hat', 'shorts']

def _load_experts(filepaths: List[str], classes_per_experts: int, device: str):
    '''Load expert models, trained each on a subset of CR_GQA

    Parameters:
    -----------
    filepaths: List[str]
        The paths to the saved models
    classes_per_expert: int
        The number of classes learned by each expert. Needed to define correctly the model classifier

    Returns:
    --------
    List[torch.Module]:
        List of experts, with loaded weights
    '''

    experts = []
    for i in range(len(filepaths)):
        # correct the classification layer
        model = resnet18()
        fc_in_features = model.fc.in_features
        model.fc = nn.Linear(fc_in_features, classes_per_experts)

        model.load_state_dict(torch.load(filepaths[i]))
        model.to(torch.device(device))
        model.eval()
        experts.append(model)

    return experts

def _dissect_image(img: torch.Tensor) -> List[torch.Tensor]:
    '''Divide the image into quadrants

    Return
    ------
    List[torch.Tensor]
        The list of quadrants, without the 'black' ones.
    '''

    quadrants = []
    im1, im2 = torch.tensor_split(img, 2, dim=1) # cut in half the image on the first dimension

    # append only non-black quadrants
    for im in (im1, im2):
        x, y = torch.tensor_split(im, 2, dim=2)
        for split in (x, y):
            if not (split < 0).all(): quadrants.append(split)

    return quadrants

def _select_expert_per_image(experts: List[nn.Module], img: torch.Tensor, divided_classes: List[np.array]) -> Union[int, str, int]:
    '''Select the apprioriate experts for the current picture.

    Specifically, select the most confident model for each quadrant.
    Ideally, appropriate experts should return the correct class, the other ones should return 4 == 'other'

    Parameters
    ----------
    experts: List[nn.Module]
        List of expert models.
    imgs: torch.Tensor
        An image represented by a tensor. It is one quadrant of an image from CGQA
    divided_classes: List[np.array]
        List of arrays, indicating the classes idxs related to each expert

    Return
    ------
    int
        The maximum confidence from an expert
    str
        The label of the predicted class
    int
        The index of the expert --- most confident model for the current quadrant
    '''

    # in the end, these variables will contain the highest confidence and the related prediction and expert model
    max_confidence, best_pred, selected_expert = float('-inf'), float('-inf'), float('-inf')

    for idx, expert in enumerate(experts):
        logits = softmax(expert(img.unsqueeze(0)).squeeze(), dim=0)
        confidence, pred = torch.max(logits, dim=0)

        #print(f'Expert {idx} predicted class {pred} with confidence {confidence}')
        
        # save the best prediction, if the class is different from 'other' == 3
        if pred != 3 and confidence > max_confidence:
            max_confidence = confidence.item()
            best_pred = divided_classes[idx][pred] # from expert class to general class idx
            selected_expert = idx

    return max_confidence, best_pred, selected_expert

def predict_composition(x: torch.Tensor, experts: List[nn.Module], divided_classes: List[np.array]) -> List:
    '''Give an image and the list of experts and the relative classes, get the prediction

    Return
    ------
    List:
        A list with two strings - the two labels that make the compositional label
    '''

    comp_prediction = []

    # get the appropriate experts for the current image in CGQA
    for quadrant in _dissect_image(x.squeeze()):
        confidence, pred, expert_idx = _select_expert_per_image(experts, quadrant, divided_classes)
        # if pred was assigned, i.e. at least an expert returned prediction label different from 'other'
        if isinstance(pred, str): comp_prediction.append(pred)
        
        #print(f'Selected expert {expert_idx}')
        #print(f'Predicted class **{classes[pred]}** with confidence {confidence}')

    #print(f'The predicted compositional label is {tuple(comp_prediction)}')
    return comp_prediction

def run(params):
    # load CGQA dataset
    dataset = cgqa.continual_training_benchmark(n_experiences=1, image_size=(196,196), fixed_class_order=list(range(100)), dataset_root='/disk3/lquara')
    test_set = dataset.test_datasets[0]

    #indices = random.choices(range(len(test_set)), k=100)
    #test_set = Subset(test_set, indices)

    # batch_size must be 1, because we need to preprocess each image and select the right experts
    loader = DataLoader(test_set, batch_size=1)

    # the classes relative to each expert
    divided_classes = np.array_split(np.array(classes), params.n_experts)

    # load models
    #model_names = ['likely-plant-12', 'prime-violet-13', 'pretty-cloud-14', 'distinctive-dew-15', 'radiant-surf-16', 'denim-shadow-17', 'warm-wildflower-18']
    #model_names = ['volcanic-forest-20', 'feasible-sponge-21', 'dauntless-meadow-22', 'denim-night-23', 'lunar-oath-24', 'rose-glade-25', 'light-donkey-26']
    model_names = ['pleasant-smoke-27', 'resilient-valley-28', 'fanciful-cosmos-29', 'ethereal-firefly-30', 'hearty-feather-31', 'twilight-rain-32', 'northern-meadow-33']
    filepaths = [params.expert_path + f'expert_{i+1}/' + name for i, name in enumerate(model_names)]
    experts = _load_experts(filepaths, classes_per_experts=params.classes_per_expert, device=params.device)

    y_true, y_pred = [], []
    # perform classification on test set
    for x, y in tqdm(loader):
        x = x.to(torch.device(params.device))
        #print(f'Actual label: {dataset.label_info[2][y.item()]}')
        comp_pred = predict_composition(x, experts, divided_classes)
        comp_pred.sort()
        #print(f'Predicted label: {comp_pred}')

        # check if the predicted label is actually in the dataset
        # otw, put the prediction to 101, which is always false
        #if tuple(comp_pred) in dataset.label_info[1].keys() or tuple(reversed(comp_pred)) in dataset.label_info[1].keys():
        if tuple(comp_pred) in dataset.label_info[1].keys():
            comp_pred_idx = dataset.label_info[1][tuple(comp_pred)]
        else: comp_pred_idx = 101

        y_true.append(y.item())
        y_pred.append(comp_pred_idx)

    # compute metrics
    y_pred = torch.Tensor(y_pred)
    y_true = torch.Tensor(y_true)
    acc = multiclass_accuracy(y_pred, y_true, num_classes=101).item()
    #auroc = multiclass_auroc(y_pred, y_true, num_classes=100)
    cm = multiclass_confusion_matrix(y_pred, y_true, num_classes=101).numpy()
    
    print(acc)
    print(cm)

    with open('results', 'w') as fp:
        fp.write(str(acc)+'\n')
        fp.write(str(cm))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_experts', type=int, default=7)
    parser.add_argument('--expert_path', type=str, default='/disk4/lquarantiello/compCV/CFST-compCV/experts/')
    parser.add_argument('--classes_per_expert', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')


    params = parser.parse_args()
    run(params)