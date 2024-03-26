import hydra
from omegaconf import OmegaConf, DictConfig
import os
import wandb

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.models import resnet18

import ds.cgqa_general as cgqa
from training import Trainer


@hydra.main(config_path='config', config_name='config')
def train_models(cfg: DictConfig):
    '''
    Return a set of models, each trained on an experience.

    Parameters
    ----------
    n_experiences : int
        The number of experiences in the benchmark
    dataset_root : str
        The root path of the dataset
    cfg: DictConfig
        The hyperparameters needed for training and evaluation, loaded using hydra and OmegaConf.
        The hyperparameters are accessible in the 'config' directory

    Returns
    -------
    models : Trained ResNets
        The set of trained models.
        Each model is trained on a single experience from the benchmark. Hence, each model is "expert" on a set of classes.
        The architecture used is ResNet18
    '''

    if cfg.log:
        wandb.init(
            project='CFST-compCV',
            config = {
                'epochs': cfg.train.epochs,
                'batch_size': cfg.train.batch_size,
                'lr': cfg.optimizer.lr,
                'momentum': cfg.optimizer.momentum,
                'nesterov': cfg.optimizer.nesterov,
                'weight_decay_lambda': cfg.optimizer.weight_decay,
                'device': cfg.train.device
            }
        )

    benchmark = cgqa.continual_training_benchmark(n_experiences=n_experiences, seed=42,
                                                  return_task_id=True, dataset_root=root)

    for exp_id in range(n_experiences):
        # get the data
        train_loader = DataLoader(benchmark.train_datasets[exp_id], batch_size=cfg.train.batch_size, shuffle=True)
        val_loader = DataLoader(benchmark.val_datasets[exp_id], batch_size=cfg.train.batch_size, shuffle=True)
        test_loader = DataLoader(benchmark.test_datasets[exp_id], batch_size=cfg.train.batch_size, shuffle=True)

        # get the model and the needed stuff for training
        model = resnet18()
        loss_fn = CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum, nesterov=cfg.optimizer.nesterov, weight_decay=cfg.optimizer.weight_decay)

        if cfg.log: save_path = os.getcwd() + wandb.run.name 
        else: save_path = os.getcwd() + '/results'

        # train and evaluate on the current experience
        trainer = Trainer(model, loss_fn=loss_fn, optimizer=optimizer, path=save_path, device=cfg.train.device, log=cfg.log)
        results = trainer.train(train_loader, val_loader, num_classes=len(benchmark.classes_in_exp[0]), epochs=cfg.train.epochs)




if __name__ == '__main__':
    train_models()