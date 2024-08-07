import os
import json
import argparse
from typing import List, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.fft as fft
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import intel_extension_for_pytorch as ipex

from my_ml.save_load_data import load_model, load_all_model_list

def get_layer_info(module: nn.Module, prefix: str = '') -> List[Dict[str, Any]]:
    """Recursively extract information about layers in the model."""
    layers_info = []
    for name, layer in module.named_children():
        layer_name = f"{prefix}.{name}" if prefix else name
        if isinstance(layer, nn.Sequential):
            layers_info.extend(get_layer_info(layer, layer_name))
        else:
            layer_info = {
                "name": layer.__class__.__name__,
                "full_name": layer_name,
                "parameters": sum(p.numel() for p in layer.parameters()),
                "dimension": get_layer_dimension(layer)
            }
            layers_info.append(layer_info)
    return layers_info

def get_layer_dimension(layer: nn.Module) -> str:
    """Get the dimension of a layer based on its type."""
    if isinstance(layer, nn.Conv2d):
        return f"{layer.in_channels}x{layer.out_channels}x{layer.kernel_size[0]}x{layer.kernel_size[1]}"
    elif isinstance(layer, nn.Linear):
        return f"{layer.in_features}x{layer.out_features}"
    elif isinstance(layer, nn.BatchNorm2d):
        return f"{layer.num_features}"
    elif isinstance(layer, nn.MaxPool2d):
        return f"kernel_size={layer.kernel_size}, stride={layer.stride}"
    elif isinstance(layer, nn.AdaptiveMaxPool2d):
        return f"output_size={layer.output_size}"
    else:
        return "N/A"

def create_truncated_model(model: nn.Module, layer_name: str) -> nn.Sequential:
    """Create a truncated model up to the specified layer."""
    layers = []
    found = False

    def add_layers(module: nn.Module, prefix: str = ''):
        nonlocal found
        for name, child in module.named_children():
            if found:
                return
            full_name = f"{prefix}.{name}" if prefix else name
            if full_name == layer_name:
                layers.append(child)
                found = True
                return
            if list(child.children()):
                add_layers(child, full_name)
            else:
                layers.append(child)
                if full_name == layer_name:
                    found = True
                    return

    add_layers(model)

    if not found:
        raise ValueError(f"Layer {layer_name} not found in the model.")

    return nn.Sequential(*layers)

def transform_input_images(input_images: torch.Tensor) -> torch.Tensor:
    """Apply normalization transform to input images."""
    transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(input_images)

class ChannelVisualizer:
    def __init__(
        self,
        model: nn.Module,
        layer_name: str,
        channel_num: int,
        input_size: Tuple[int, int] = (1, 28, 28),
        num_images: int = 4,
        num_iterations: int = 500,
        learning_rate: float = 1e-2,
        activation_weight: float = 1.0,
        diversity_weight: float = 0,
        regularization_weight: float = 0.0,
        total_variance_weight: float = 0.0
    ):
        self.model = model
        self.layer_name = layer_name
        self.channel_num = channel_num
        self.input_size = input_size
        self.num_images = num_images
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

        self.truncated_model = create_truncated_model(model, layer_name)
        self.truncated_model.eval()

        self.device = next(model.parameters()).device  # Get the device of the model
        self.input_images_raw = torch.randn((num_images, *input_size), requires_grad=True, device=self.device)
        self.optimizer = torch.optim.Adam([self.input_images_raw], lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)

        # self.model, self.optimizer = ipex.optimize(self.model, optimizer=self.optimizer)

        self.activation_weight = torch.tensor(activation_weight, device=self.device, requires_grad=False)
        self.diversity_weight = torch.tensor(diversity_weight, device=self.device, requires_grad=False)
        self.regularization_weight = torch.tensor(regularization_weight, device=self.device, requires_grad=False)
        self.total_variance_weight = torch.tensor(total_variance_weight, device=self.device, requires_grad=False)

    def visualize(self) -> Tuple[List[Image.Image], List[List[Tuple[int, Image.Image]]], Dict[str, List[float]]]:
        """Perform the channel visualization process."""
        best_loss = float('inf')
        patience = 0
        patience_limit = 1024
        min_improvement = 1e-2
        keep_training = True

        intermediate_images = [[] for _ in range(self.num_images)]
        save_epochs = [0] + [2 ** i for i in range(1, int(np.log2(self.num_iterations)) + 1)]
        losses = {
            'total_loss': [],
            'activation_loss': [],
            'l2_reg': [],
            'tv_reg': [],
            'div_loss': []
        }

        for epoch in tqdm(range(self.num_iterations)):
            loss, input_images = self._train_step()

            losses['total_loss'].append(loss.item())
            losses['activation_loss'].append(self.activation_loss.item())
            losses['l2_reg'].append(self.l2_reg.item())
            losses['tv_reg'].append(self.tv_reg.item())
            losses['div_loss'].append(self.div_loss.item())

            if loss.item() < best_loss - min_improvement:
                best_loss = loss.item()
                patience = 0
            else:
                patience += 1
                if patience > patience_limit:
                    keep_training = False

            if epoch in save_epochs or epoch == self.num_iterations - 1 or not keep_training:
                self._save_intermediate_images(input_images, epoch, intermediate_images)

            if not keep_training:
                break

        final_images = self._get_final_images(input_images)
        return final_images, intermediate_images, losses

    @staticmethod
    def apply_transformations(images: torch.Tensor, num_transforms: int = 4) -> torch.Tensor:
        transformed_images = []
        for image in images:
            image_transforms = [image]
            for _ in range(num_transforms - 1):  # -1 because we already have the original image
                transformed = transforms.Compose([
                    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=15),
                    transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
                ])(image)
                image_transforms.append(transformed)
            transformed_images.extend(image_transforms)
        return torch.stack(transformed_images)

    def _train_step(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.optimizer.zero_grad()

        input_images = transform_input_images(self.input_images_raw)

        # Apply transformations
        transformed_images = self.apply_transformations(input_images)

        x = self.truncated_model(transformed_images)
        target: torch.Tensor = x[:, self.channel_num]

        # Calculate individual losses
        activation_loss = -torch.mean(target) * self.activation_weight
        l2_reg = torch.norm(input_images) * self.regularization_weight
        tv_reg = ((torch.sum(torch.abs(input_images[:, :, :, :-1] - input_images[:, :, :, 1:])) +
                   torch.sum(torch.abs(input_images[:, :, :-1, :] - input_images[:, :, 1:, :])))
                  * self.total_variance_weight)

        if self.num_images > 1:
            flat_images = self.input_images_raw.view(self.num_images, -1)
            similarity = torch.mm(flat_images, flat_images.t())
            div_loss = ((similarity.sum() - similarity.diag().sum()) / (
                    self.num_images * (self.num_images - 1))) * self.diversity_weight
        else:
            div_loss = torch.tensor(0.0, device=input_images.device, requires_grad=True)

        # Compute total loss
        total_loss = activation_loss + l2_reg + tv_reg + div_loss

        # Compute gradient
        total_loss.backward()

        # Update the parameters
        self.optimizer.step()

        # Store individual losses for logging
        self.activation_loss = activation_loss.detach()
        self.l2_reg = l2_reg.detach()
        self.tv_reg = tv_reg.detach()
        self.div_loss = div_loss.detach()

        # clamp the input images to be in the range [0, 1]
        self.input_images_raw.data.clamp_(0, 1)

        return total_loss.detach(), input_images

    def _save_intermediate_images(self, input_images: torch.Tensor, epoch: int, intermediate_images: List[List[Tuple[int, Image.Image]]]):
        """Save intermediate images during the visualization process."""
        for j in range(self.num_images):
            intermediate_image = input_images[j].squeeze().detach().cpu().numpy()
            intermediate_image = (intermediate_image - intermediate_image.min()) / (intermediate_image.max() - intermediate_image.min() + 1e-5)
            intermediate_images[j].append((epoch, Image.fromarray((intermediate_image * 255).astype(np.uint8))))

    def _get_final_images(self, input_images: torch.Tensor) -> List[Image.Image]:
        """Get the final images after the visualization process."""
        final_images = []
        for j in range(self.num_images):
            result = input_images[j].squeeze().detach().cpu().numpy()
            result = (result - result.min()) / (result.max() - result.min() + 1e-5)
            final_images.append(Image.fromarray((result * 255).astype(np.uint8)))
        return final_images

def save_visualization_results(
    layer_name: str,
    channel_num: int,
    intermediate_images: List[List[Tuple[int, Image.Image]]],
    directory: str = "layers"
) -> None:
    """Save visualization results for a specific channel."""
    num_images = len(intermediate_images)
    num_iterations = len(intermediate_images[0])

    fig, axes = plt.subplots(num_images, num_iterations,
                             figsize=(4 * num_iterations, 4 * num_images))
    fig.suptitle(f"Visualization Process of {layer_name} channel {channel_num}", fontsize=16)

    for i in range(num_images):
        for j in range(num_iterations):
            epoch, img = intermediate_images[i][j]
            ax = axes[i, j] if num_images > 1 and num_iterations > 1 else axes[i] if num_images > 1 else axes[j]
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Image {i + 1}, Epoch {epoch}')
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"channel_{channel_num}.png"))
    plt.close()

def aggregate_and_save_layer_images(
    layer_name: str,
    all_channel_images: List[List[Image.Image]],
    directory: str = "layers"
):
    """Aggregate and save images for all channels in a layer."""
    num_channels = len(all_channel_images)
    num_images_per_channel = len(all_channel_images[0])

    print(f"plotting {num_images_per_channel} images for {num_channels} channels")

    fig, axes = plt.subplots(num_images_per_channel, num_channels,
                             figsize=(4 * num_channels, 4 * num_images_per_channel))
    fig.suptitle(f"Aggregated Channel Visualizations for {layer_name}", fontsize=16)

    for i in range(num_images_per_channel):
        for j in range(num_channels):
            ax = axes[i, j] if num_images_per_channel > 1 else axes[j]
            ax.imshow(all_channel_images[j][i], cmap='gray', aspect='auto')
            ax.set_title(f'Channel {j}, Image {i + 1}')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(directory, "aggregated_channels.png"), dpi=300, bbox_inches='tight')
    plt.close()


def save_training_losses(
        layer_name: str,
        channel_num: int,
        losses: Dict[str, List[float]],
        directory: str = 'layers',
        focus_last_n: int = 128
) -> None:
    """Save training loss plots for a specific channel."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 15))
    fig.suptitle(f"Training Losses for {layer_name} channel {channel_num}")

    colors = plt.cm.rainbow(np.linspace(0, 1, len(losses)))

    def plot_losses(ax, data, title):
        y_min, y_max = float('inf'), float('-inf')
        for (loss_name, loss_values), color in zip(data.items(), colors):
            ax.plot(loss_values, label=loss_name, color=color)
            y_min = min(y_min, min(loss_values))
            y_max = max(y_max, max(loss_values))

        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.2)

    # Plot entire training process
    plot_losses(ax1, losses, "Full Training")

    # Plot last N iterations
    last_n_losses = {k: v[-focus_last_n:] for k, v in losses.items()}
    plot_losses(ax2, last_n_losses, f"Last {focus_last_n} iterations")
    ax2.set_xlim(len(next(iter(losses.values()))) - focus_last_n, len(next(iter(losses.values()))))

    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"channel_{channel_num}_training_loss.png"), dpi=300)
    plt.close()


def main(model_name: str = "MNIST_crude_model"):
    model = load_model(model_name)

    all_layers_info = get_layer_info(model)

    save_root = os.path.join("layers", model_name)

    num_iterations = 2 ** 12
    learning_rate = 1e-3
    num_images = 1

    activation_weight = 1
    diversity_weight = 1e1
    regularization_weight = 5e-3
    total_variance_weight = 3e-4

    os.makedirs(save_root, exist_ok=True)

    with open(os.path.join(save_root, "all_layers_info.json"), "wt") as f:
        json.dump(all_layers_info, f, indent=4)

    for layer_info in all_layers_info:
        layer_name = layer_info["full_name"]

        if layer_info["name"] == "Conv2d":
            channel_cnt = int(layer_info["dimension"].split('x')[1])

            channel_save_root = os.path.join(save_root, layer_name.replace('.', '_'))
            channel_vis_path = os.path.join(channel_save_root, "channel_images")
            channel_loss_path = os.path.join(channel_save_root, "losses")

            os.makedirs(channel_save_root, exist_ok=True)
            os.makedirs(channel_vis_path, exist_ok=True)
            os.makedirs(channel_loss_path, exist_ok=True)

            all_channel_images = []

            for channel_num in range(channel_cnt):
                print(f"Visualizing layer {layer_name}, channel {channel_num}...")

                visualizer = ChannelVisualizer(
                    model,
                    layer_name,
                    channel_num,
                    num_images=num_images,
                    num_iterations=num_iterations,
                    learning_rate=learning_rate,

                    activation_weight=activation_weight,
                    diversity_weight=diversity_weight,
                    regularization_weight=regularization_weight,
                    total_variance_weight=total_variance_weight
                )
                final_images, intermediate_images, losses = visualizer.visualize()

                save_visualization_results(
                    layer_name,
                    channel_num,
                    intermediate_images,
                    directory=channel_vis_path
                )
                all_channel_images.append(final_images)

                save_training_losses(
                    layer_name,
                    channel_num,
                    losses,
                    directory=channel_loss_path,
                    # focus_last_n=num_iterations // 8
                )

            # Aggregate and save images for the entire layer
            aggregate_and_save_layer_images(layer_name, all_channel_images, directory=channel_save_root)

    print(f"Visualization complete. Results saved in the '{save_root}' folder.")


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default=None)

    inp = args.parse_args().model

    if inp not in load_all_model_list():
        print("\n".join(load_all_model_list()))
        inp = input("Enter the model name: ")

    trained_model = load_model(inp)

    save_root = os.path.join('layers', inp, 'filters')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    main(inp)