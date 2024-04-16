'''Load expert models trained on CR_GQA, and use them zero-shot on CGQA test set
'''

import argparse
import numpy as np
from typing import List

import torch
from torch import nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from ds import cgqa_general as cgqa

# the classes from CR_GQA, derived following the instruction from CGQA/continual json files
classes = ['plate', 'shirt', 'building', 'sign', 'grass', 'car', 'table', 'chair', 'jacket', 'shoe', 'flower', 'pants', 'helmet', 'bench', 'pole', 'leaves', 'wall', 'door', 'fence', 'hat', 'shorts']

def _load_experts(filepaths: List[str], classes_per_experts: int):
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

def _select_experts_per_image(experts: List[nn.Module], img: torch.Tensor) -> int:
    '''Select the apprioriate experts for the current picture.

    Specifically, select the most confident model for each quadrant.
    Ideally, appropriate experts should return the correct class, the other ones should return 4 == 'other'

    Parameters
    ----------
    experts: List[nn.Module]
        List of expert models.
    imgs: torch.Tensor
        An image represented by a tensor. It is one quadrant of an image from CGQA

    Return
    ------
    int
        The index of the expert --- most confident model for the current quadrant
    '''

    # in the end, these variables will contain the highest confidence and the related prediction and expert model
    max_confidence, best_pred, selected_expert = float('-inf'), float('-inf'), float('-inf')

    for idx, expert in enumerate(experts):
        logits = softmax(expert(img.unsqueeze(0)).squeeze(), dim=0)
        confidence, pred = torch.max(logits, dim=0)

        if confidence > max_confidence:
            max_confidence = confidence
            best_pred = pred
            selected_expert = idx

    return max_confidence, best_pred, idx

def predict_composition(x: torch.Tensor, experts: List[nn.Module]):
    '''Give an image and the list of experts, get the prediction
    '''

    # get the appropriate experts for the current image in CGQA
    current_experts: List[int] = []
    for quadrant in _dissect_image(x.squeeze()):
        confidence, pred, expert_idx = _select_experts_per_image(experts, quadrant)
        
        print(f'Selected expert {expert_idx}')
        print(f'Predicted class **{classes[pred]}** with confidence {confidence}')

    # possibly, you need to use avalanche benchmark when loading cgqa
    # otherwise, I think you cannot retrieve the class label from the index
    # also, it would be nice to print the image without transform - just to check by eye that everything is correct

    return

def test_experts(experts: nn.Module, test_loader: DataLoader):
    pass

def run(params):
    # load CGQA dataset
    dataset = cgqa.continual_training_benchmark(n_experiences=1, dataset_root='/disk3/lquara')
    test_set = dataset.test_datasets[0]
    # batch_size must be 1, because we need to preprocess each image and select the right experts
    test_loader = DataLoader(test_set, batch_size=1)

    # the classes relative to each expert
    divided_classes = np.array_split(np.array(classes), params.n_experts)

    # load models
    model_names = ['likely-plant-12', 'prime-violet-13', 'pretty-cloud-14', 'distinctive-dew-15', 'radiant-surf-16', 'denim-shadow-17', 'warm-wildflower-18']
    filepaths = [params.expert_path + f'expert_{i+1}/' + name for i,name in enumerate(model_names)]
    experts = _load_experts(filepaths, classes_per_experts=params.classes_per_expert)

    for x, y in test_loader:
        pred = predict_composition(x, experts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_experts', type=int, default=7)
    parser.add_argument('--expert_path', type=str, default='/disk4/lquarantiello/compCV/CFST-compCV/experts/')
    parser.add_argument('--classes_per_expert', type=int, default=4)


    params = parser.parse_args()
    run(params)