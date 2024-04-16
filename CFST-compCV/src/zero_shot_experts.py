'''Load expert models trained on CR_GQA, and use them zero-shot on CGQA test set
'''

import argparse
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from ds import cgqa_general as cgqa

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

def _select_experts_per_image(experts: List[nn.Module], imgs: List[torch.Tensor]) -> List[nn.Module]:
    '''Select the apprioriate experts for the current picture.

    Specifically, select an expert per quadrant.
    Ideally, appropriate experts should return the correct class, the other ones should return 4 == 'other'

    Parameters
    ----------
    experts: List[nn.Module]
        List of expert models.
    imgs: List[torch.Tensor]
        List of images, which are the quadrants of one sample from CGQA.

    Return
    ------
    List[nn.Module]
        List of selected experts.
    '''

    pass

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

def test_experts(experts: nn.Module, test_loader: DataLoader):
    pass

def run(params):
    # load CGQA dataset
    dataset = cgqa.continual_training_benchmark(n_experiences=1, dataset_root='/disk3/lquara')
    test_set = dataset.test_sets[0]
    # batch_size must be 1, because we need to preprocess each image and select the right experts
    test_loader = DataLoader(test_set, batch_size=1)

    # load models
    filepaths = ['likely-plant-12', 'prime-violet-13', 'pretty-cloud-14', 'distinctive-dew-15', 'radiant-surf-16', 'denim-shadow-17', 'warm-wildflower-18']
    experts = _load_experts(filepaths, classes_per_experts=params.classes_per_expert)

    # cut pictures into 4 quadrants

    for x, y in test_loader:
        exps = _select_experts_per_image(experts, _dissect_image(y))



    results = test_experts(experts, test_loader)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_experts', type=int, default=7)
    parser.add_argument('--expert_path', type=str, default='/disk4/lquarantiello/compCV/CFST-compCV/experts/')
    parser.add_argument('--classes_per_expert', type=int, default=4)


    params = parser.parse_args()
    run(params)