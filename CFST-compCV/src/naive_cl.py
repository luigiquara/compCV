# from https://github.com/NTU-LANTERN/CFST/tree/main
import argparse
from datetime import datetime

import torch
from torch.nn import CrossEntropyLoss
import avalanche as avl
from avalanche.training import JointTraining, Naive
from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin, LRSchedulerPlugin
from avalanche.training.checkpoint import save_checkpoint
from avalanche.evaluation import metrics as metrics

from ds import cgqa
from models.resnet import get_resnet

def get_evaluation_plugin(model, wandb_logger):
    wandb_logger.wandb.watch(model)
    loggers = [
        avl.logging.InteractiveLogger(), 
        wandb_logger
    ]

    evaluation_plugin = EvaluationPlugin(
        *metrics_list,
        # benchmark=benchmark,
        loggers=loggers)

    return evaluation_plugin


parser = argparse.ArgumentParser()
parser.add_argument('--strategy', type=str)
parser.add_argument('--taskIL', type=int)
param = parser.parse_args()

if param.strategy == 'naive':
    n_experiences = 10
elif param.strategy == 'joint':
    n_experiences = 1

# if task-incremental, num_classes/n_experiences features for each head; in our case, = 10
# if class-incremental, all features all at once, = 100 classes
if param.taskIL: initial_out_features = 10
elif param.taskIL: initial_out_features = 100

benchmark = cgqa.continual_training_benchmark(n_experiences=n_experiences, seed=1234, return_task_id=param.taskIL, dataset_root='/disk3/lquara')

# Define resnet18 with multiple heads for task-IL setting
model = get_resnet(multi_head=param.taskIL, initial_out_features=initial_out_features)
breakpoint()

# Define metrics to use
metrics_list = [
    metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
    metrics.loss_metrics(epoch=True, experience=True, stream=True),
    #metrics.forgetting_metrics(experience=True, stream=True),
    #metrics.forward_transfer_metrics(experience=True, stream=True),
    metrics.timing_metrics(epoch=True)]

# Define naive strategy
optimizer = torch.optim.Adam(model.parameters(), lr=8e-3)
epochs = 1
plugins = [
    LRSchedulerPlugin(scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-6)), 
    EarlyStoppingPlugin(patience=5, val_stream_name='val_stream')
]


# define the logger and evaluation plugin
run_name = param.strategy + '_taskIL' if param.taskIL else '_classIL'
wandb_logger = avl.logging.WandBLogger(project_name='CFST-compCV', run_name=run_name, config={'benchmark':'cgqa'})
wandb_logger.wandb.watch(model)
loggers = [
    avl.logging.InteractiveLogger(), 
    wandb_logger
]
evaluation_plugin = EvaluationPlugin(
    *metrics_list,
    # benchmark=benchmark,
    loggers=loggers)


if param.strategy == 'naive':
    strategy = Naive(model, optimizer, CrossEntropyLoss(),
                     train_mb_size=100, train_epochs=epochs, eval_mb_size=50, device=torch.device('cuda'),
                     plugins=plugins, evaluator=evaluation_plugin, eval_every=1, peval_mode="epoch")
elif param.strategy == 'joint':
    strategy = JointTraining(model, optimizer, CrossEntropyLoss(),
                             train_mb_size=100, train_epochs=epochs, eval_mb_size=50,
                             device=torch.device('cuda'), plugins=plugins, evaluator=evaluation_plugin, eval_every=1)
    
# Start training 
results = []
for exp_idx, (experience, val_task) in enumerate(zip(benchmark.train_stream, benchmark.val_stream)):
    print("Start of experience ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)
    print("Current Classes (str): ", [
        benchmark.label_info[2][cls_idx]
        for cls_idx in benchmark.original_classes_in_exp[experience.current_experience]
    ])

    strategy.train(experience, eval_streams=[val_task])
    print("Training experience completed")

    print("Computing accuracy on the whole test set.")
    result = strategy.eval(benchmark.test_stream)
    results.append(result)

print(results)
#save_checkpoint(strategy, fname=datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + '.strategy')
torch.save(model.state_dict(), datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + '.strategy')

import code; code.interact(local=locals())