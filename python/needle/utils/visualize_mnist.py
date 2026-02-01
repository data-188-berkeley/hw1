from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_lfw_people


def plot_image_gallery(images: list[np.ndarray], titles: list[str], n_row: int = 3, n_col: int = 4) -> Figure:
    """Plots images+titles to figure.

    Args:
        images (list[np.ndarray]): shape=[num_images, img_height, img_width]
        titles (list[str]): len=num_images.
        n_row (int, optional): Number of rows in figure. Defaults to 3.
        n_col (int, optional): Number of cols in figure. Defaults to 4.

    Returns:
        Figure: fig_out.
    """
    # inspired by:
    #   https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html
    # Helper function to plot a gallery of portraits
    if len(images) != len(titles):
        raise RuntimeError(f"num images must match num text titles: {len(images)} vs {len(titles)}")
    fig = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    fig.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        axes_subplot = fig.add_subplot(n_row, n_col, i + 1)

        axes_subplot.imshow(images[i], cmap=plt.cm.gray)
        axes_subplot.set_title(titles[i], size=12)
        axes_subplot.set_xticks(())
        axes_subplot.set_yticks(())
    return fig


def visualize_mnist_epoch(pred_logits: np.ndarray, X: np.ndarray, num_chans_img: int, h_img: int, w_img: int, y: np.ndarray, digit: int = 4, top_k: int = 5):
    """Visualizes the predictions of an MNIST digit classifier.

    Args:
        pred_logits (np.ndarray): Model predictions. shape=[num_samples, num_classes].
        X (np.ndaray): Test images. shape=[num_samples, num_pixels], arranged row-by-row.
            Images are 28x28.
        num_chans_img: Number of image channels. Should be 1, these are grayscale images.
        h_img: Image height, in pixels.
        w_img: Image width, in pixels.
        y (np.ndarray): Test labels. shape=[num_samples].
        digit: Which ground-truth digit to visualize predictions for.
        top_k: How many examples to visualize.

    Returns:
        fig: matplotlib Figure instance.
    """
    # shape=[num_samples, num_classes]
    pred_probs = _softmax_normalize(pred_logits)
    pred_labels = pred_probs.argmax(axis=1)
    mask_correct_preds = pred_labels == y

    # Visualize top k most confident true positives
    tp_indices = (mask_correct_preds == 1) & (y == digit)
    tp_probs = pred_probs[tp_indices, digit]

    sorted_inds = np.argsort(tp_probs)[::-1]
    sorted_inds_top_k = sorted_inds[:top_k]
    top_tp_probs = tp_probs[sorted_inds_top_k]

    Xtp = X[tp_indices, :]
    Xviz = Xtp[sorted_inds_top_k, :]

    yviz = y[tp_indices][sorted_inds_top_k]

    if num_chans_img > 1:
        images = Xviz.reshape([Xviz.shape[0], num_chans_img, h_img, w_img])
    else:
        images = Xviz.reshape([Xviz.shape[0], h_img, w_img])

    titles = [f"p={top_tp_probs[i]:.3f} y_gt={yviz[i]}" for i in range(len(top_tp_probs))]

    fig_tp = plot_image_gallery(images, titles, n_row=1, n_col=5)
    fig_tp.suptitle(f"Top {top_k} most confident true positives (digit={digit})")

    # Visualize top k least confident false negatives ("hard" examples)
    # Sort false negatives by their predicted prob for `digit`.
    # Take top k lowest scores, and visualize them.
    fn_indices = (mask_correct_preds == 0) & (y == digit)

    fn_probs = pred_probs[fn_indices, digit]

    sorted_inds_top_k = np.argsort(fn_probs)[:top_k]
    top_fn_probs = fn_probs[sorted_inds_top_k]

    # what did we predict?
    pred_label_fn = pred_labels[fn_indices][sorted_inds_top_k]
    pred_prob_fn = pred_probs[fn_indices, :][sorted_inds_top_k, pred_label_fn]

    Xviz = X[fn_indices, :][sorted_inds_top_k, :]
    yviz = y[fn_indices][sorted_inds_top_k]

    if num_chans_img > 1:
        images = Xviz.reshape([Xviz.shape[0], num_chans_img, h_img, w_img])
    else:
        images = Xviz.reshape([Xviz.shape[0], h_img, w_img])

    titles = [f"\np_gt={top_fn_probs[i]:.3f} y_gt={yviz[i]}\npred={pred_label_fn[i]} ({pred_prob_fn[i]:.3f})" for i in range(len(top_fn_probs))]

    fig_fn = plot_image_gallery(images, titles, n_row=1, n_col=5)
    fig_fn.suptitle(f"Top {top_k} least confident false negatives (digit={digit})\n")

    # make room for suptitle
    fig_fn.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.8, hspace=0.35)
    return fig_tp, fig_fn


def _softmax_normalize(vals: np.ndarray) -> np.ndarray:
    """Perform softmax normalization of input vals. The normalization
    is done across the last dimension.
    Args:
        vals: shape=[d0, d1, ..., dim].
    Returns:
        vals_norm: shape=[d0, d1, ..., dim].
    """
    vals_exp = np.exp(vals)
    norm_factor = np.sum(vals_exp, axis=-1)
    # note: need `norm_factor[:, np.newaxis]` to broadcast the division
    #   across the axis=1 of vals_exp
    #   otherwise, this line produces an error, due to numpy not knowing
    #   how to divide an array with shape=[batchsize, k] with an array with
    #   shape=[k].
    #     vals_exp / norm_factor
    return vals_exp / (norm_factor[:, np.newaxis])


def demo_mnist():
    # Demo: download LFW face dataset, plot first 9 people, then save figure to disk
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    _, h, w = lfw_people.images.shape

    images = lfw_people.images[:9, :, :]
    titles = lfw_people.target_names[lfw_people.target[:9]]

    fig = plot_image_gallery(images, titles, n_row=3, n_col=3)
    fig.savefig("./lfw_faces.png")


if __name__ == '__main__':
    demo_mnist()
