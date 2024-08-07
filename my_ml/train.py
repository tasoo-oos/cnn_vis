import os
from torch import nn, optim
import argparse
import torch
from typing import Dict, Any
import intel_extension_for_pytorch as ipex

from .dataset import load_data
from .model_class import CNN, DenseNet, ResNet
from .train_loop import evaluate, train
from .save_load_data import save_model, load_training_settings
from .plotting import plot_training_history
from .callbacks import EarlyStopping, ModelCheckpoint, LRScheduler
from .config import DATASET_CONFIGS, MODEL_CONFIGS, DATASET_TO_MODEL

def get_model_config(dataset_name: str) -> Dict[str, Any]:
    model_type = DATASET_TO_MODEL[dataset_name]
    config = {**DATASET_CONFIGS[dataset_name], **MODEL_CONFIGS[model_type]}
    return config

def get_model(dataset_name: str, model_config: Dict[str, Any]) -> nn.Module:
    model_type = DATASET_TO_MODEL[dataset_name]
    if model_type == 'CNN':
        return CNN(**model_config)
    elif model_type == 'DenseNet':
        return DenseNet(**model_config)
    elif model_type == 'ResNet':
        return ResNet(**model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def main(dataset_name: str,
         model_name: str,
         train_from_middle: bool = False
         ) -> None:
    # Ensure necessary directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not train_from_middle:
        model_config = get_model_config(dataset_name)
        model = get_model(dataset_name, model_config)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        training_data = None
        last_epoch = -1
    else:
        # Load the model and its training history
        training_data = load_training_settings(model_name)
        dataset_name = training_data['dataset_name']
        model_config = training_data['model_config']
        model = training_data['model']
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.load_state_dict(training_data['optimizer_state'])
        last_epoch = len(training_data['history']['train_loss']) - 1

    model = model.to(device)
    model, optimizer = ipex.optimize(model, optimizer=optimizer)

    # Initialize criterion
    criterion = nn.CrossEntropyLoss()

    # Load data
    train_loader, test_loader = load_data(dataset_name)

    # Train model
    callbacks = [
        # EarlyStopping(patience=10),
        ModelCheckpoint(f'models/{model_name}_temp.pth'),
        LRScheduler(torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1, last_epoch=last_epoch))
    ]

    best_model, history = train(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        callbacks=callbacks,
        prev_training_history=training_data['history'] if training_data else None,
        device=device
    )

    print("Training completed.")

    model.load_state_dict(best_model)
    final_loss, final_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Final test loss: {final_loss:.4f}")
    print(f"Final test accuracy: {final_accuracy:.4f}")

    # Save the model
    save_model(
        model_name,
        dataset_name,
        model_config,
        model,
        history,
        optimizer
    )

    # plot the training history
    plot_training_history(model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the specified dataset.")
    parser.add_argument("--dataset",
                        type=str,
                        default="MNIST",
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Name of the dataset to train the model on.")
    parser.add_argument("--model_desc",
                        type=str,
                        default="simple_model",
                        help="Message to display during training.")
    parser.add_argument("--load_model",
                        action="store_true",
                        help="Load a previously trained model instead of training a new one.")

    args = parser.parse_args()

    model_name = f'{args.dataset}_{args.model_desc}'
    print(f"{'Loading' if args.load_model else 'Training'} model {model_name}...")
    main(args.dataset, model_name, args.load_model)
