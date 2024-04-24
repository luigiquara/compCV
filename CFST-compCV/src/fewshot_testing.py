from collections import OrderedDict

import avalanche as avl
from avalanche.evaluation import metrics
from avalanche.training import EvaluationPlugin, LRSchedulerPlugin

from ds import cgqa
from models.resnet import get_resnet

def _adjust_state_dict(model_path: str) -> OrderedDict:
    sd = torch.load(model_path)
    del sd['classifier.active_units']
    del sd['classifier.classifier.weight']
    del sd['classifier.classifier.bias']

    return sd


def run():
    benchmark = cgqa.fewshot_testing_benchmark(n_experiences=300, mode='sys', dataset_root='/disk3/lquara')

    loggers = [avl.logging.InteractiveLogger()]
    metrics_list = [metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
                    metrics.loss_metrics(epoch=True, experience=True, stream=True)]

    evaluation_plugin = EvaluationPlugin(*metrics_list, loggers=loggers)

    # load pretrained model
    # replay, classIL, pretrained on CR_GQA
    model = get_resnet(multi_head=False, initial_out_features=10)
    model.resnet.load_state_dict(_adjust_state_dict('ckpt/replay_classIL_191418.ckpt'))

if __name__ == '__main__':
    run()