import hydra
from omegaconf import DictConfig, OmegaConf
import time
import wandb

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torchvision import models
from torchvision import transforms

import ds.cgqa_general as cgqa

class LResNet(L.LightningModule):
    def __init__(self, resnet_name, num_classes, optimizer_cfg):
        super().__init__()
        self.resnet = self._get_resnet(resnet_name, num_classes)
        self.num_classes = num_classes

        self.loss_fn = CrossEntropyLoss()
        self.optimizer_cfg = optimizer_cfg

        self.save_hyperparameters()

    def configure_optimizers(self):
        if self.optimizer_cfg.name == 'adam':
            optimizer = Adam(self.resnet.parameters(), lr=self.optimizer_cfg.lr, weight_decay=self.optimizer_cfg.weight_decay)
        elif self.optimizer_cfg.name == 'sgd':
            optimizer = SGD(self.resnet.parameters(), lr=self.optimizer_cfg.lr, momentum=self.optimizer_cfg.momentum, nesterov=self.optimizer_cfg.nesterov, weight_decay=self.optimizer_cfg.weight_decay)
        
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=50)

        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }

    def training_step(self, batch, batch_idx):
        _, loss, acc = self._get_preds_loss_acc(batch)

        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, loss, acc = self._get_preds_loss_acc(batch)

        self.log('val_loss', loss)
        self.log('val_acc', acc)
    
    def test_step(self, batch, batch_idx):
        _, loss, acc = self._get_preds_loss_acc(batch)

        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def _get_preds_loss_acc(self, batch):
        x, y = batch
        preds = self.resnet(x)
        loss = self.loss_fn(preds, y)
        acc = accuracy(preds, y, task='multiclass', num_classes=self.num_classes)
        return preds, loss, acc
    
    def _get_resnet(self, resnet_name, num_classes):
        if resnet_name == 'resnet18':
            resnet = models.resnet18(progress=True, num_classes=num_classes)
        elif resnet_name == 'resnet34':
            resnet = models.resnet34(progress=True, num_classes=num_classes)
        elif resnet_name == 'resnet50':
            resnet = models.resnet50(progress=True, num_classes=num_classes)
        elif resnet_name == 'resnet101':
            resnet = models.resnet101(progress=True, num_classes=num_classes)
        elif resnet_name == 'resnet152':
            resnet = models.resnet152(progress=True, num_classes=num_classes)
        return resnet

@hydra.main(config_path='config', config_name='config', version_base=None)
def run(cfg: DictConfig):

    # preprocessing transformations
    # from CFST appendix
    tr_preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    benchmark = cgqa.continual_training_benchmark(n_experiences=cfg.n_experiences, seed=42, return_task_id=True, dataset_root='/disk3/lquara/', train_transform=tr_preprocess, eval_transform=test_preprocess)

    # train a different model for each experience
    for exp_id in range(cfg.n_experiences):
        # wandb + lightning
        wandb_logger = WandbLogger(project='CFST-compCV', log_model='all', group=f'exp {exp_id}')
        wandb_logger.experiment.config['benchmark'] = 'cgqa'

        # define the model
        num_classes = len(benchmark.classes_in_exp[0])
        model = LResNet(cfg.model_name, num_classes, cfg.optimizer)

        # get the data
        train_loader = DataLoader(benchmark.train_datasets[exp_id], batch_size=cfg.train.batch_size, shuffle=True, num_workers=11)
        val_loader = DataLoader(benchmark.val_datasets[exp_id], batch_size=cfg.train.batch_size, num_workers=11)
        test_loader = DataLoader(benchmark.test_datasets[exp_id], batch_size=cfg.train.batch_size, num_workers=11)

        # train the model
        early_stopping_cb = EarlyStopping(monitor='val_loss', mode='min', patience=5)
        checkpoint_cb = ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=1)
        trainer = L.Trainer(callbacks=[early_stopping_cb, checkpoint_cb], logger = wandb_logger)
        trainer.fit(model, train_loader, val_loader)

        wandb.finish()

        break

if __name__ == '__main__':
    run()