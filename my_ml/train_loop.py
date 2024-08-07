import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any, List
from tqdm import tqdm

from .callbacks import Callback, ModelCheckpoint

def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct / total


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer,
                device: torch.device) -> Tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_accuracy = 0.0
    for data, targets in tqdm(train_loader, desc="Training"):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        scores = model(data)
        loss = criterion(scores, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_accuracy += calculate_accuracy(scores, targets)
    return total_loss / len(train_loader), total_accuracy / len(train_loader)


def evaluate(model: nn.Module,
             data_loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for data, targets in tqdm(data_loader, desc="Evaluating"):
            data, targets = data.to(device), targets.to(device)
            scores = model(data)
            loss = criterion(scores, targets)
            total_loss += loss.item()
            total_accuracy += calculate_accuracy(scores, targets)
    return total_loss / len(data_loader), total_accuracy / len(data_loader)


def train(model: nn.Module,
          train_loader: DataLoader,
          test_loader: DataLoader,
          criterion: nn.Module,
          optimizer: optim.Optimizer,
          max_epochs: int = 1024,
          callbacks: List[Callback] = None,
          prev_training_history: Dict[str, List[float]] = None,
          device: torch.device = None
          ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[float]]]:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = criterion.to(device)

    if callbacks is None:
        callbacks = []

    if prev_training_history is None:
        train_loss_history, test_loss_history, train_acc_history, test_acc_history = [], [], [], []
        start_epoch = 0
    else:
        train_loss_history = prev_training_history['train_loss']
        test_loss_history = prev_training_history['test_loss']
        train_acc_history = prev_training_history['train_acc']
        test_acc_history = prev_training_history['test_acc']
        start_epoch = len(train_loss_history)

    logs = {}
    for callback in callbacks:
        callback.on_train_begin(logs)

    try:
        for epoch in range(start_epoch, max_epochs):
            for callback in callbacks:
                callback.on_epoch_begin(epoch, logs)

            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            train_loss_history.append(train_loss)
            test_loss_history.append(test_loss)
            train_acc_history.append(train_acc)
            test_acc_history.append(test_acc)

            logs = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'model': model
            }

            print(
                f'Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

            stop_training = False
            for callback in callbacks:
                if callback.on_epoch_end(epoch, logs):
                    stop_training = True

            if stop_training:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    except KeyboardInterrupt:
        print('Training interrupted by user.')

    for callback in callbacks:
        callback.on_train_end(logs)

    # Load the best model if a ModelCheckpoint callback was used
    model_checkpoint = next((cb for cb in callbacks if isinstance(cb, ModelCheckpoint)), None)
    if model_checkpoint:
        best_state_dict = model_checkpoint.get_best_model()
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            print("Loaded the best model from checkpoint.")

    training_history = {
        'train_loss': train_loss_history,
        'test_loss': test_loss_history,
        'train_acc': train_acc_history,
        'test_acc': test_acc_history
    }

    return model.state_dict(), training_history