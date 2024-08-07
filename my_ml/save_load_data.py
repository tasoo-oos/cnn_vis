from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Type

import dill
import torch
from torch import nn

def save_model(
    model_name: str,
    dataset_name: str,
    model_config: Dict[str, Any],
    model: nn.Module,
    history: Dict[str, List[float]],
    optimizer: torch.optim.Optimizer,
    directory: str = 'models'
) -> None:
    """Save a trained model and its training history."""

    model_data = {
        'model_name': model_name,
        'model_dataset': dataset_name,
        'model_architecture': {
            'class': model.__class__,
            'model_config': model_config,
        },
        'model_param': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'history': history
    }

    file_path = Path(directory) / f"{model_name}.dill"
    try:
        with open(file_path, 'wb') as f:
            dill.dump(model_data, f)
    except IOError as e:
        print(f"Error saving model: {e}")

def load_model(model_name: str, directory: str = 'models') -> Optional[nn.Module]:
    """Load a trained model from the specified directory."""
    file_path = Path(directory) / f"{model_name}.dill"
    try:
        return initialize_model(file_path)
    except (IOError, KeyError, dill.UnpicklingError) as e:
        print(f"Error loading model: {e}")
        raise

def initialize_model(file_path: Path) -> nn.Module:
    with open(file_path, 'rb') as f:
        data = dill.load(f)
    model_class = data['model_architecture']['class']
    model_config = data['model_architecture']['model_config']
    model = model_class(**model_config)
    model.load_state_dict(data['model_param'])
    return model

def load_training_history(model_name: str, directory: str = 'models') -> Optional[Dict[str, List[float]]]:
    """Load the training history of a model from the specified directory."""
    file_path = Path(directory) / f"{model_name}.dill"
    try:
        with open(file_path, 'rb') as f:
            data = dill.load(f)
        return data['history']
    except (IOError, KeyError, dill.UnpicklingError) as e:
        print(f"Error loading training history: {e}")
        return None

def load_training_settings(model_name: str, directory: str = 'models') -> Dict[str, Any]:
    """Load the training settings of a model from the specified directory."""
    file_path = Path(directory) / f"{model_name}.dill"
    try:
        with open(file_path, 'rb') as f:
            data = dill.load(f)

        model = initialize_model(file_path)

        return {
            'model': model,
            'model_name': data['model_name'],
            'dataset_name': data['model_dataset'],
            'model_config': data['model_architecture']['model_config'],
            'history': data['history'],
            'optimizer_state': data['optimizer_state']
        }
    except (IOError, KeyError, dill.UnpicklingError) as e:
        print(f"Error loading training settings: {e}")
        raise

def load_all_model_list(directory: str = 'models') -> List[str]:
    """Return a list of all model names in the specified directory."""
    return [str(file).split("/")[-1].split(".")[0] for file in Path(directory).glob("*.dill")]

if __name__ == '__main__':
    print("\n".join(load_all_model_list()))

    inp = input("Enter the model name: ")
    model = load_model(inp)

    if model is not None:
        from torchsummary import summary
        input_size = (1, 28, 28)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        summary(model, input_size)

        from torchviz import make_dot
        make_dot(model(torch.randn(1, *input_size).to(device)), params=dict(model.named_parameters())).render(f"results/{inp}_model_architecture", format="png")

        from .plotting import plot_training_history
        plot_training_history(inp)
