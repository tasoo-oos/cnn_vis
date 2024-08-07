import torch

class Callback:
    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass


class EarlyStopping(Callback):
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current = logs['test_loss']
        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True
        return False

class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='test_loss', save_best_only=True, save_to_file=False, verbose=False):
        self.filepath = filepath
        self.best = float('inf')
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_to_file = save_to_file
        self.verbose = verbose
        self.best_state_dict = None

    def on_epoch_end(self, epoch, logs=None):
        current = logs[self.monitor]
        if current < self.best:
            self.best = current
            self.best_state_dict = logs['model'].state_dict().copy()
            if self.save_to_file:
                torch.save(self.best_state_dict, self.filepath)
            if self.verbose:
                print(f'Epoch {epoch+1}, Model saved to {self.filepath}')

    def get_best_model(self):
        return self.best_state_dict


class LRScheduler(Callback):
    def __init__(self, scheduler, verbose=False):
        self.scheduler = scheduler
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 0:
            self.scheduler.step()
            if self.verbose:
                print(f'Epoch {epoch+1}, Learning Rate: {self.scheduler.get_last_lr()}')