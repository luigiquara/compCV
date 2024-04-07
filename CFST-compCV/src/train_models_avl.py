import argparse
from datetime import datetime

from avalanche.evaluation import metrics
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.training import JointTraining, Naive, Replay
from avalanche.training.plugins import EarlyStoppingPlugin, EvaluationPlugin, LRSchedulerPlugin
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from ds import cgqa
from models.resnet import get_resnet

def create_strategy(strategy_name, model, optimizer, plugins, evaluator, params: argparse.Namespace):

    if strategy_name == 'naive':
        strategy = Naive(model, optimizer, CrossEntropyLoss(), train_epochs=params.epochs,
                        train_mb_size=params.train_mb_size, eval_mb_size=params.eval_mb_size,
                        device=torch.device(params.device), plugins=plugins, evaluator=evaluator,
                        eval_every=1, peval_mode='epoch')
    elif strategy_name == 'multitask':
        strategy = JointTraining(model, optimizer, CrossEntropyLoss(), train_epochs=params.epochs,
                        train_mb_size=params.train_mb_size, eval_mb_size=params.eval_mb_size,
                        device=torch.device(params.device), plugins=plugins, evaluator=evaluator,
                        eval_every=1)
    elif strategy_name == 'replay':
        strategy = Replay(model, optimizer, CrossEntropyLoss(), train_epochs=params.epochs,
                        train_mb_size=params.train_mb_size, eval_mb_size=params.eval_mb_size,
                        mem_size=params.mem_size, device=torch.device(params.device),
                        plugins=plugins, evaluator=evaluator, eval_every=1, peval_mode='epoch')

    return strategy

def _joint_train(strategy, benchmark):
    results = []
    strategy.train(benchmark.train_stream, eval=[benchmark.val_stream])
    print('Ended training on the entire stream')
    results.append(strategy.eval(benchmark.test_stream))
    
    return results

def _continual_train(strategy, benchmark):
    results = []
    for exp_idx, (experience, val_task) in enumerate(zip(benchmark.train_stream, benchmark.val_stream)):
        print("Start of experience ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        print("Current Classes (str): ", [benchmark.label_info[2][cls_idx] for cls_idx in benchmark.original_classes_in_exp[experience.current_experience]])

        strategy.train(experience, eval_streams=[val_task])
        print("Training experience completed")

        print("Computing accuracy on the whole test set.")
        results.append(strategy.eval(benchmark.test_stream))
    return results

def train(strategy, benchmark):
    if isinstance(strategy, JointTraining): results = _joint_train(strategy, benchmark)
    else: results = _continual_train(strategy, benchmark)
    return results

def run(params):
    # get benchmark and model
    benchmark = cgqa.continual_training_benchmark(n_experiences=10, seed=1234, return_task_id=params.is_taskIL, dataset_root='/disk3/lquara')
    model = get_resnet(multi_head=params.is_taskIL, initial_out_features=10)

    # define training plugins
    optimizer = Adam(model.parameters(), lr=params.lr)
    plugins = [
        LRSchedulerPlugin(scheduler=CosineAnnealingLR(optimizer, params.epochs, 1e-6)), 
        EarlyStoppingPlugin(patience=5, val_stream_name='val_stream')
    ]

    # define the evalution plugin, with metrics and loggers
    metrics_list = [metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
                    metrics.loss_metrics(epoch=True, experience=True, stream=True),
                    metrics.timing_metrics(epoch=True)]
    loggers = [InteractiveLogger()]
    if params.use_wandb:
        run_name = params.strategy + '_taskIL' if params.is_taskIL else params.strategy + '_classIL'
        wandb_logger = WandBLogger(project_name='CFST-compCV', run_name=run_name, config={'benchmark':'cgqa'})
        wandb_logger.wandb.watch(model)
        loggers.append(wandb_logger)

    evaluation_plugin = EvaluationPlugin(*metrics_list, loggers = loggers)
    
    strategy = create_strategy(params.strategy, model, optimizer, plugins=plugins,
                               evaluator=evaluation_plugin, params=params)
    results = train(strategy, benchmark)
    if params.save_model:
        setting = 'taskIL' if params.is_taskIL else 'classIL'
        filename = 'ckpt/'+params.strategy+'_'+setting+'_'+datetime.now().strftime('%H%M%S')+'.ckpt'
        torch.save(strategy.model.state_dict(), filename)
        print(f'Model saved at {filename}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str)
    parser.add_argument('--is_taskIL', action='store_true', dest='is_taskIL')
    parser.add_argument('--is_classIL', action='store_false', dest='is_taskIL')
    parser.add_argument('--lr', type=float, default=8e-3)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--no-use_wandb', action='store_false', dest='use_wandb')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--no-save_model', action='store_false', dest='save_model')

    # strategy arguments
    parser.add_argument('--train_mb_size', type=int, default=100)
    parser.add_argument('--eval_mb_size', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--mem_size', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')

    params = parser.parse_args()
    run(params)