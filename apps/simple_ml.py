"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys
import time

import needle as ndl


def parse_mnist(image_filename: str, label_filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def train_nn_two_layer_mnist(
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_te: np.ndarray,
        y_te: np.ndarray,
        hidden_dim: int = 500,
        epochs: int = 10,
        lr: float = 0.5,
        batch: int = 100,
        visualize_preds: bool = False
    ) -> tuple[tuple[ndl.Tensor, ndl.Tensor], dict]:
    """Trains a two-layer neural network on MNIST digit images (multi-class classification via cross-entropy loss).

    Args:
        X_tr (np.ndarray): Training dataset. shape=[n_train, dim_input]
        y_tr (np.ndarray): Training labels. shape=[n_train]
        X_te (np.ndarray): Test dataset. shape=[n_test, dim_input]
        y_te (np.ndarray): Test labels. shape=[n_test]
        hidden_dim (int, optional): Hidden dim of NN. Defaults to 500.
        epochs (int, optional): Number of train epochs. Defaults to 10.
        lr (float, optional): Learning rate. Defaults to 0.5.
        batch (int, optional): Batchsize. Defaults to 100.
        visualize_preds (bool, optional): If True, visualize predictions. Defaults to False.
    Returns:
        model_params: (W1, W2)
        train_meta: Training metadata. Things like: train/test loss, etc.
    """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = ndl.Tensor(np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim))
    W2 = ndl.Tensor(np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k))

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err | secs |")
    # For fun, print train/test loss at very start of training (with randomly-init'd params, epoch=-1)
    # Important: use `Tensor.detach()` so that these calculations aren't added to the computation graph
    train_loss, train_err = loss_err(ndl.relu(ndl.Tensor(X_tr) @ W1.detach()) @ W2.detach(), y_tr)
    test_loss, test_err = loss_err(ndl.relu(ndl.Tensor(X_te) @ W1.detach()) @ W2.detach(), y_te)
    print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} | {:.2f} |"\
            .format(-1, train_loss, train_err, test_loss, test_err, -1))

    train_meta = {}

    train_losses, train_errs = [train_loss], [train_err]
    test_losses, test_errs = [test_loss], [test_err]
    epoch_durs = []
    for epoch in range(epochs):
        tic_epoch = time.time()
        W1, W2 = nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(ndl.relu(ndl.Tensor(X_tr) @ W1.detach()) @ W2.detach(), y_tr)
        test_loss, test_err = loss_err(ndl.relu(ndl.Tensor(X_te) @ W1.detach()) @ W2.detach(), y_te)
        dur_epoch = time.time() - tic_epoch

        train_losses.append(train_loss)
        train_errs.append(train_err)
        test_losses.append(test_loss)
        test_errs.append(test_err)
        epoch_durs.append(dur_epoch)

        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} | {:.2f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err, dur_epoch))

    train_meta["train_losses"] = train_losses
    train_meta["train_errs"] = train_errs
    train_meta["test_losses"] = test_losses
    train_meta["test_errs"] = test_errs
    train_meta["epoch_durs"] = epoch_durs

    if visualize_preds:
        # Visualize model predictions
        from needle.utils.visualize_mnist import visualize_mnist_epoch
        final_pred_logits = ndl.relu(ndl.Tensor(X_te) @ W1.detach()) @ W2.detach()
        for digit in range(0, 10):
            visualize_mnist_epoch(final_pred_logits.numpy(), X_te, 1, 28, 28, y_te, digit=digit, top_k=5)

    return (W1, W2), train_meta



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h: ndl.Tensor, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Helper function to compute both loss and error
    Args:
        h: input predicted logits. shape=[batchsize, num_classes]
        y: ground-truth labels. shape=[batchsize]
    Returns:
        softmax_loss: Softmax (cross-entropy) loss. Averaged.
        cls_error: Classification error (eg zero-one error). Averaged.
    """
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
