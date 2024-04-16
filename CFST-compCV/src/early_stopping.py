import torch

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.counter = 0
        self.patience = patience
        self.min_delta = min_delta
        self.min_validation_loss = float('inf')

    def stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            print('My patience is wearing thin...')
            if self.counter > self.patience: return True
        
        return False


class Saver:
    def __init__(self, path):
        self.path = path
        self.best_model = None
        self.min_validation_loss = float('inf')

    def update_best_model(self, model, validation_loss):
        if validation_loss < self.min_validation_loss:
            print('Hey, new best model in town!')
            self.min_validation_loss = validation_loss
            self.best_model = model

    def save(self):
        print('Saving the best model')
        torch.save(self.best_model.state_dict(), self.path)