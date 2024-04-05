import argparse
from datetime import datetime

import avalanche as avl
from avalanche.models.utils import avalanche_model_adaptation
from avalanche.training import Replay
from avalanche.training.checkpoint import save_checkpoint
from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin, LRSchedulerPlugin
from avalanche.evaluation import metrics as metrics
import torch
from torch.nn import CrossEntropyLoss

from ds import cgqa
from models.resnet import get_resnet

def create_strategy(is_taskIL, use_wandb=True):
    model = get_resnet(multi_head=is_taskIL, initial_out_features=10)

    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4)
    epochs = 100
    plugins = [
        LRSchedulerPlugin(scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-6)), 
        EarlyStoppingPlugin(patience=5, val_stream_name='val_stream')
    ]

    metrics_list = [
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        metrics.loss_metrics(epoch=True, experience=True, stream=True),
        #metrics.forgetting_metrics(experience=True, stream=True),
        #metrics.forward_transfer_metrics(experience=True, stream=True),
        metrics.timing_metrics(epoch=True)
    ]

    loggers = [avl.logging.InteractiveLogger()]
    if use_wandb:
        run_name = 'replay' + '_taskIL' if is_taskIL else '_classIL'
        wandb_logger = avl.logging.WandBLogger(project_name='CFST-compCV', run_name=run_name, config={'benchmark':'cgqa'})
        wandb_logger.wandb.watch(model)
        loggers.append(wandb_logger)

    evaluation_plugin = EvaluationPlugin(*metrics_list,loggers = loggers)# benchmark=benchmark


    strategy = Replay(model, optimizer, CrossEntropyLoss(), mem_size=1000,
                      train_mb_size=100, train_epochs=epochs, eval_mb_size=50,
                      device=torch.device('cuda'), plugins=plugins, evaluator=evaluation_plugin, eval_every=1)
    return strategy

def run(param):
    benchmark = cgqa.continual_training_benchmark(n_experiences=10, seed=1234, return_task_id=param.taskIL, dataset_root='/disk3/lquara')

    strategy = create_strategy(param.taskIL) 
    results = []
    strategy.train(benchmark.train_stream, eval_streams=[benchmark.train_stream, benchmark.val_stream])
    print('Ended training on the entire stream')
    results.append(strategy.eval(benchmark.test_stream))
    #save_checkpoint(strategy, fname=datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + '.strategy')
    torch.save(strategy.model.state_dict(), 'replay_'+datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + '.ckpt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--taskIL', type=int)
    param = parser.parse_args()

    run(param)