import matplotlib.pyplot as plt
from .save_load_data import load_training_history


def plot_training_history(model_name: str) -> None:
    """Plot the training and test loss and accuracy. Data is read from the 'models' directory."""

    data = load_training_history(model_name)

    train_loss_history = data['train_loss']
    test_loss_history = data['test_loss']
    train_acc_history = data['train_acc']
    test_acc_history = data['test_acc']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    fig.suptitle(f'{model_name} Training History', fontsize=16, fontweight='bold')

    # Loss plot
    ax1.plot(train_loss_history, label='Train Loss', color='blue', linestyle='-', marker='o', markersize=4)
    ax1.plot(test_loss_history, label='Test Loss', color='red', linestyle='-', marker='s', markersize=4)
    ax1.set_title('Training and Test Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # Accuracy plot
    ax2.plot(train_acc_history, label='Train Accuracy', color='green', linestyle='-', marker='o', markersize=4)
    ax2.plot(test_acc_history, label='Test Accuracy', color='purple', linestyle='-', marker='s', markersize=4)
    ax2.set_title('Training and Test Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(f'results/{model_name}_train_history.png', dpi=300)
    plt.close()
    print(f"Training history plot saved to results/{model_name}_train_history.png")


if __name__ == '__main__':
    plot_training_history('MNIST_test')