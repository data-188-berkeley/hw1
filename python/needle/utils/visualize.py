from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


def plot_train_test_curves(
        losses_train: np.ndarray,
        errors_train: np.ndarray,
        losses_test: np.ndarray,
        errors_test: np.ndarray,
        ) -> Figure:
    """Plot train/test loss+error.

    Args:
        losses_train (np.ndarray): Train loss, after each epoch/step.
        errors_train (np.ndarray): Train error, after each epoch/step.
        losses_test (np.ndarray): Test loss, after each epoch/step.
        errors_test (np.ndarray): Test error, after each epoch/step.

    Returns:
        Figure: figure.
    """
    epochs = range(0, len(losses_train))

    # Plotting the loss curves
    fig, (ax_loss, ax_error) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax_loss.plot(epochs, losses_train, 'r', linewidth=1, marker='o', label='Training loss')
    ax_loss.plot(epochs, losses_test, 'b', linewidth=1, marker='o', label='Test loss')

    # Add details to the plot
    ax_loss.set_title('Training/Test Loss', fontsize=12)
    ax_loss.set_xlabel('Epochs', fontsize=10)
    ax_loss.set_ylabel('Loss', fontsize=10)
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True)

    # Plot error curves
    ax_error.plot(epochs, errors_train, 'r', linewidth=1, marker='o', label='Training error')
    ax_error.plot(epochs, errors_test, 'b', linewidth=1, marker='o', label='Test error')

    # Add details to the plot
    ax_error.set_title('Training/Test Error', fontsize=12)
    ax_error.set_xlabel('Epochs', fontsize=10)
    ax_error.set_ylabel('Error', fontsize=10)
    ax_error.legend(fontsize=8)
    ax_error.grid(True)

    return fig
