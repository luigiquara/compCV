import time
from tqdm import tqdm
import numpy as np
from collections import defaultdict

import torch
from torch import nn
from torch.nn.functional import normalize

from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassConfusionMatrix
from sklearn.metrics import balanced_accuracy_score

from early_stopping import Saver, EarlyStopper

import wandb

class Trainer:

    def __init__(self, model, loss_fn=None, optimizer=None, current_classes=None, save_path=None, device='cpu', log=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.current_classes = current_classes
        self.device = torch.device('cuda') if device=='cuda' and torch.cuda.is_available() else torch.device('cpu')
        self.log = log

        self.stopper = EarlyStopper(patience=3, min_delta=0)
        self.saver = Saver(save_path)

        print('Initialized trainer')
        print(f'Using {self.loss_fn}')
        print(f'Using {self.optimizer}')
        print(f'Using {self.device}')

    def preprocess_labels(self, y):
        '''Preprocess the labels, for the (1 vs All)-classification

        The label for the current classes (the ones that the current expert has to learn) remain the same.
        All the other labels are changed to 'other'
        '''
        for i, class_idx in enumerate(y):
            if class_idx not in self.current_classes:
                y[i] = 21 # 'other' == 21

        # normalize class indexes
        y = [3 if x == 21 else self.current_classes.index(x) for x in y]
        return torch.tensor(y, dtype=int)

    def train(self, tr_loader, val_loader, num_classes, epochs=5):
        self.model.to(self.device)

        tr_results = defaultdict(list)
        val_results = defaultdict(list)
        
        tr_mca = MulticlassAccuracy(num_classes=num_classes, average='macro').to(self.device)
        tr_auroc = MulticlassAUROC(num_classes=num_classes, average='macro').to(self.device)
        tr_cm = MulticlassConfusionMatrix(num_classes=num_classes).to(self.device)
        val_mca = MulticlassAccuracy(num_classes=num_classes, average='macro').to(self.device)
        val_auroc = MulticlassAUROC(num_classes=num_classes, average='macro').to(self.device)
        val_cm = MulticlassConfusionMatrix(num_classes=num_classes).to(self.device)
        
        for e in range(epochs):
            print(f'Epoch {e+1}/{epochs}')

            self.model.train()
            tr_results = self.training_step(tr_loader, [tr_mca, tr_auroc, tr_cm])
            tr_results['epoch'] = e

            print(f'Training accuracy: {tr_results["training_accuracy"].item()}')
            print(f'Training loss: {tr_results["training_loss"]}')
            if self.log: wandb.log(tr_results)

            print('Validating...')
            self.model.eval()
            val_results = self.eval_step(val_loader, [val_mca, val_auroc, val_cm])
            val_results['epoch'] = e

            print(f'Validation Accuracy: {val_results["validation_accuracy"].item()}')
            print(f'Validation loss: {val_results["validation_loss"]}')
            if self.log:
                wandb.log(val_results)
                f, ax = val_cm.plot(add_text=True)
                f.set_figwidth(7)
                f.set_figheight(7)
                wandb.log({"Validation Confusion Matrix": wandb.Image(f)})

            # saving and early stopping
            self.saver.update_best_model(self.model, val_results['validation_loss'])
            if self.stopper.stop(val_results['validation_loss']):
                print(f'Stopping at epoch {e+1}')
                break
        self.saver.save()

        return tr_results, val_results

    def training_step(self, tr_loader, tr_metrics):
        losses = [] 
        res = {}

        for X, old_y in tqdm(tr_loader):
            self.optimizer.zero_grad()

            X = X.to(self.device)

            y = self.preprocess_labels(old_y)
            y = y.to(self.device)

            y_pred = self.model(X)

            loss = self.loss_fn(y_pred, y)
            loss.backward()
            self.optimizer.step()

            #if len(mb.shape) != 3: # one-hot encoded input
            #    mb_labels = torch.argmax(mb, dim=1) # from one-hot encoding to labels
            #else: mb_labels = mb

            # logging
            for m in tr_metrics:
                m.update(y_pred, y)
            losses.append(loss.item())

        for m in tr_metrics:
            name = 'training_' + str(m).split('(')[0].split('Multiclass')[1].lower()
            res[name] = m.compute()
            m.reset()

        # average loss per item
        res['training_loss'] = sum(losses) / (len(tr_loader.dataset))
        return res

    def eval_step(self, val_loader, val_metrics):
        with torch.no_grad():
            total_loss = 0
            res = {}
            for m in val_metrics: m.reset()

            for X, y in tqdm(val_loader):
                X = X.to(self.device)

                y = self.preprocess_labels(y)
                y = y.to(self.device)

                y_pred = self.model(X)
                loss = self.loss_fn(y_pred, y)
                total_loss += loss.item()

                #logging
                #if len(mb.shape) != 3: # one-hot encoded input
                #    mb_labels = torch.argmax(mb, dim=1) # from one-hot encoding to labels
                #    for m in vl_metrics:
                #        m.update(logits, mb_labels)
                #else:
                #    for m in vl_metrics:
                #        m.update(logits, mb)

                for m in val_metrics:
                    m.update(y_pred, y)

            for m in val_metrics:
                name = 'validation_' + str(m).split('(')[0].split('Multiclass')[1].lower()
                res[name] = m.compute()

            # average of the loss for a single item
            res['validation_loss'] = total_loss / (len(val_loader.dataset))
        return res 

    def test(self, test_set):
        self.model.eval()

        test_results = defaultdict(list)
        test_mca = MulticlassAccuracy(num_classes=num_classes)

        test_loss, test_acc = self.eval_step(test_set, test_mca)
        test_results['test_loss'].append(test_loss)
        test_results['test_accuracy'].append(test_acc)

        return test_results

    def forward_pass(self, frame):
        self.model.eval()
        logits = self.model(frame.to(self.device))
        return logits

    def load_model(self, model_weights):
        self.model.load_state_dict(torch.load(model_weights))
        self.model.to(self.device)