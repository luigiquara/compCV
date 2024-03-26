import hydra
from omegaconf import DictConfig, OmegaConf

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchmetrics.functional import accuracy
from torchvision import models

import ds.cgqa_general as cgqa

class LResNet(L.LightningModule):
    def __init__(self, resnet_name, lr, momentum, nesterov):
        super().__init__()
        self.resnet = self._get_resnet(resnet_name)
        self.loss = CrossEntropyLoss()
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = SGD(self.resnet.parameters(), lr=self.lr, momentum=self.momentum, nesterov=self.nesterov)
        return optimizer

    def training_step(self, batch, batch_idx):
        _, loss, acc = self._get_preds_loss_acc(batch)

        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, loss, acc = self._get_preds_loss_acc(batch)

        self.log('val_loss', loss)
        self.log('val_acc', acc)
    
    def test_set(self, batch, batch_idx):
        _, loss, acc = self._get_preds_loss_acc(batch)

        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def _get_preds_loss_acc(self, batch):
        x, y = batch
        preds = self.resnet(x)
        loss = self.loss_fn(preds, y)
        acc = accuracy(preds, y)
        return preds, loss, acc
    
    def _get_resnet(self, resnet_name):
        if resnet_name == 'resnet18':
            resnet = models.resnet18(progress=True)
        elif resnet_name == 'resnet34':
            resnet = models.resnet34(progress=True)
        elif resnet_name == 'resnet50':
            resnet = models.resnet50(progress=True)
        elif resnet_name == 'resnet101':
            resnet = models.resnet101(progress=True)
        elif resnet_name == 'resnet152':
            resnet = models.resnet152(progress=True)
        return resnet

@hydra.main(config_path='config', config_name='config', version_base=None)
def run(cfg: DictConfig):
    benchmark = cgqa.continual_training_benchmark(n_experiences=cfg.n_experiences, seed=42,
                                                  return_task_id=True, dataset_root='/disk3/lquara/')
    # train a different model for each experience
    for experience in range(cfg.n_experiences):
        # wandb + lightning
        wandb_logger = WandbLogger(project = 'CFST-compCV', log_model='all')

        # define the model
        model = LResNet(cfg.model_name, cfg.optimizer.lr, cfg.optimizer.momentum, cfg.optimizer.nesterov)

        # get the data
        train_loader = DataLoader(benchmark.train_datasets[exp_id], batch_size=cfg.train.batch_size, shuffle=True)
        val_loader = DataLoader(benchmark.val_datasets[exp_id], batch_size=cfg.train.batch_size, shuffle=True)
        test_loader = DataLoader(benchmark.test_datasets[exp_id], batch_size=cfg.train.batch_size, shuffle=True)

        # train the model
        trainer = L.Trainer(callbacks=[EarlyStopping(monitor='val_loss', mode='min')], logger = WandbLogger)
        trainer.fit(model, train_loader, val_loader)

        break

if __name__ == '__main__':
    run()