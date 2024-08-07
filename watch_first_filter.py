from my_ml.save_load_data import load_model, load_all_model_list

import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_filters(model, save_root):
    for i, module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d) and module.in_channels in [1, 3]:
            # Get the weights of the layer
            weights = module.weight.data.cpu()

            # Normalize the weights
            min_val = weights.min()
            max_val = weights.max()
            weights = (weights - min_val) / (max_val - min_val)

            # Create a grid of subplots
            num_filters = weights.shape[0]
            num_rows = int(np.ceil(np.sqrt(num_filters)))
            num_cols = int(np.ceil(num_filters / num_rows))

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
            fig.subplots_adjust(hspace=0.1, wspace=0.1)

            for j, ax in enumerate(axes.flat):
                if j < num_filters:
                    # If the filter is for a grayscale image (1 channel)
                    if weights.shape[1] == 1:
                        img = weights[j, 0]
                    # If the filter is for a color image (3 channels)
                    if weights.shape[1] == 3:
                        img = weights[j].permute(1, 2, 0)  # Change from CxHxW to HxWxC

                    ax.imshow(img, cmap='gray' if weights.shape[1] == 1 else None)
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_root, f'filter_visualization_layer_{i}.png'), dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Filter visualization saved to {f'layers/filter_visualization_layer_{i}.png'}")


if __name__ == '__main__':
    print("\n".join(load_all_model_list()))
    inp = input("Enter the model name: ")
    trained_model = load_model(inp)

    save_root = os.path.join('layers', inp, 'filters')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    visualize_filters(trained_model, save_root)