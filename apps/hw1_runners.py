import argparse
import time

"""
Example usage:
    $ python apps/hw1_runners.py
"""


def demo_train_nn_two_layer():
    # Train a two-layer neural network on MNIST digit classification dataset.
    from apps.simple_ml import train_nn_two_layer_mnist, parse_mnist
    from needle.utils.visualize import plot_train_test_curves
    import matplotlib.pyplot as plt

    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                            "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                            "data/t10k-labels-idx1-ubyte.gz")
    tic0 = time.time()
    (W1, W2), train_meta = train_nn_two_layer_mnist(
        X_tr, y_tr, X_te, y_te,
        hidden_dim=400,
        epochs=3,
        lr=0.2,
        batch=100,
        visualize_preds=False
    )
    dur_train = time.time() - tic0
    print(f"Finished training ({dur_train:.4f}secs)")
    print(train_meta)
    fig = plot_train_test_curves(train_meta["train_losses"], train_meta["train_errs"], train_meta["test_losses"], train_meta["test_errs"])
    plt.show()


def main():
    demo_train_nn_two_layer()


if __name__ == '__main__':
    main()
