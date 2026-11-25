"""
Deep Autoencoders (2006) from Scratch in NumPy scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Deep Autoencoders from Scratch in NumPy (Hinton and Salakhutdinov, 2006).

Story: pretrain a stack of RBMs on 20,000 MNIST digits (784-500-250-125-30, the
top RBM with linear Gaussian code units), unroll it into an eight-layer
autoencoder and fine-tune it with back-propagation of the cross-entropy. Train
the same network from small random weights for the same number of epochs, and
fit PCA with 30 components. Compare the squared reconstruction error per test
image. Then repeat with a 2-unit code and measure how well the 2-D codes keep
the ten digit classes apart, against the first two principal components.
Takes a few minutes on the site's CPU.
"""
import numpy as np


def load_mnist_pixels(n_train, n_test, seed=0):
    """A random subset of MNIST as the paper feeds it to its autoencoders: (x_train, y_train, x_test, y_test).
    Each image is flattened to 784 pixel intensities divided by 255, so 0.0 is the white background and 1.0 is
    full ink (float64 arrays of shape (n, 784)); labels are int64 digits 0-9, used only to measure how well the
    codes separate the classes. The same seed always gives the same subset."""
    import dml_data
    ds = dml_data.load("mnist")
    rng = np.random.default_rng(seed)
    tr = rng.permutation(len(ds.y_train))[:n_train]
    te = rng.permutation(len(ds.y_test))[:n_test]
    x_train = ds.x_train[tr].reshape(n_train, 784).astype(np.float64) / 255.0
    x_test = ds.x_test[te].reshape(n_test, 784).astype(np.float64) / 255.0
    return x_train, ds.y_train[tr].astype(np.int64), x_test, ds.y_test[te].astype(np.int64)


def main():
    N_TRAIN, N_TEST = 20000, 1000
    SIZES = [784, 500, 250, 125, 30]
    RBM_EPOCHS, FT_EPOCHS = 2, 6

    # ---- 1. The data ----
    x, y, xt, yt = load_mnist_pixels(N_TRAIN, N_TEST, seed=0)
    print(f"[1] MNIST subset: {N_TRAIN} training and {N_TEST} test digits, 784 intensities in [0, 1]")
    print(f"    pixels below 0.1: {(x < 0.1).mean():.2f}, above 0.9: {(x > 0.9).mean():.2f}, in between: {((x >= 0.1) & (x <= 0.9)).mean():.2f}")
    mean_err = float(((xt - x.mean(axis=0)) ** 2).sum(axis=1).mean())
    print(f"    squared error per test image when every digit is 'reconstructed' as the average image: {mean_err:.2f}")

    # ---- 2. Pretraining ----
    rng = np.random.default_rng(0)
    stack, rbm_errors = pretrain_stack(x, SIZES, RBM_EPOCHS, rng)
    print(f"\n[2] greedy pretraining of {len(stack)} RBMs, {RBM_EPOCHS} passes each (the paper: 50), mini-batches of 100")
    for (W, b_vis, b_hid), errs in zip(stack, rbm_errors):
        kind = "linear Gaussian code units, lr 0.001" if W is stack[-1][0] else "binary hidden units, lr 0.1"
        print(f"    RBM {W.shape[0]:>3}-{W.shape[1]:<3} ({kind}): reconstruction error per vector {errs[0]:.2f} -> {errs[-1]:.2f}")

    # ---- 3. Unrolling ----
    layers = unroll(stack)
    n_params = sum(W.size + b.size for W, b in layers)
    widths = [layers[0][0].shape[0]] + [W.shape[1] for W, b in layers]
    print(f"\n[3] unrolled autoencoder {'-'.join(str(w) for w in widths)}: {len(layers)} weight matrices, {n_params:,} parameters")
    start = reconstruction_errors(autoencoder_forward(layers, xt)[-1], xt)[1]
    print(f"    squared error per test image before fine-tuning: {start:.2f}")

    # ---- 4. Fine-tuning: pretrained against random ----
    print(f"\n[4] fine-tuning with back-propagation of the cross-entropy, {FT_EPOCHS} epochs, lr 0.02, momentum 0.9")
    hist_pre = finetune(layers, x, xt, FT_EPOCHS, np.random.default_rng(1))
    print("    pretrained, test error after each epoch:  " + " ".join(f"{h:6.2f}" for h in hist_pre))
    random_layers = random_autoencoder(SIZES, np.random.default_rng(2))
    r0 = reconstruction_errors(autoencoder_forward(random_layers, xt)[-1], xt)[1]
    hist_rand = finetune(random_layers, x, xt, FT_EPOCHS, np.random.default_rng(1))
    print(f"    random N(0, 0.1) start ({r0:.2f} before):  " + " ".join(f"{h:6.2f}" for h in hist_rand))
    out = autoencoder_forward(random_layers, xt)[-1]
    print(f"    the random network's reconstructions differ from image to image by {out.std(axis=0).mean():.3f} per pixel on average "
          f"(the test images themselves: {xt.std(axis=0).mean():.3f})")

    # ---- 5. PCA ----
    mean, comps = fit_pca(x, SIZES[-1])
    pca_err = float(((pca_reconstruct(xt, mean, comps) - xt) ** 2).sum(axis=1).mean())
    print(f"\n[5] squared error per test image with {SIZES[-1]}-dimensional codes")
    print("    method                         this run   paper (all 60,000 digits, 200 epochs)")
    print(f"    deep autoencoder, pretrained   {hist_pre[-1]:8.2f}   3.00")
    print(f"    same net, random start         {hist_rand[-1]:8.2f}   (not reported for MNIST)")
    # logistic PCA as the paper's supporting material fits it: linear code units connected directly to the
    # inputs and to logistic output units, trained on the cross-entropy. With no hidden layers it needs a smaller
    # learning rate than the deep net (0.02 makes it diverge), so it gets 0.005.
    lpca = random_autoencoder([784, SIZES[-1]], np.random.default_rng(4))
    hist_lpca = finetune(lpca, x, xt, FT_EPOCHS, np.random.default_rng(1), lr=0.005)
    print(f"    logistic PCA (784-30-784)      {hist_lpca[-1]:8.2f}   8.01")
    print(f"    PCA                            {pca_err:8.2f}   13.87")
    print(f"    average image, for reference   {mean_err:8.2f}")
    print("    In the paper the fully trained deep autoencoder (3.00) beats logistic PCA (8.01) and PCA (13.87).")
    print(f"    Here, after {RBM_EPOCHS} pretraining and {FT_EPOCHS} fine-tuning epochs on {N_TRAIN:,} digits, it beats PCA but not")
    print("    logistic PCA. The cause is the training budget: logistic PCA has one layer of 23,520 weights on each side and")
    print("    is nearly converged in a few epochs, while the deep net's 1.1 million parameters are still improving every")
    print("    epoch (see [4]). To close the gap, raise FT_EPOCHS and RBM_EPOCHS towards the paper's 200 and 50, train on")
    print("    more digits, or use the paper's 784-1000-500-250-30 sizes, if you have the time to run it.")
    print("    the first test digit, its autoencoder reconstruction and its PCA reconstruction:")
    rec_ae = autoencoder_forward(layers, xt[:1])[-1][0]
    rec_pca = pca_reconstruct(xt[:1], mean, comps)[0]
    for a, b, c in zip(xt[0].reshape(28, 28)[4:24], rec_ae.reshape(28, 28)[4:24], rec_pca.reshape(28, 28)[4:24]):
        row = lambda r: "".join("#" if v > 0.5 else ("+" if v > 0.2 else ".") for v in r[4:24])
        print(f"      {row(a)}   {row(b)}   {row(c)}")

    # ---- 6. Two-dimensional codes ----
    features = x
    for W, b_vis, b_hid in stack[:-1]:
        features = sigmoid(features @ W + b_hid)
    W2, bv2, bh2, err2 = train_rbm(features, 2, RBM_EPOCHS, rng, linear_hidden=True, lr=0.001)
    layers2 = unroll(stack[:-1] + [(W2, bv2, bh2)])
    hist2 = finetune(layers2, x, xt, FT_EPOCHS, np.random.default_rng(3))
    code = len(layers2) // 2
    n_ref = 5000
    z_train = autoencoder_forward(layers2, x[:n_ref])[code]
    z_test = autoencoder_forward(layers2, xt)[code]
    mean2, comps2 = fit_pca(x, 2)
    p_train = (x[:n_ref] - mean2) @ comps2.T
    p_test = (xt - mean2) @ comps2.T
    pca2_err = float(((pca_reconstruct(xt, mean2, comps2) - xt) ** 2).sum(axis=1).mean())
    acc_ae = knn_accuracy(z_train, y[:n_ref], z_test, yt)
    acc_pca = knn_accuracy(p_train, y[:n_ref], p_test, yt)
    print(f"\n[6] 2-D codes: the same three pretrained RBMs, a new top RBM 125-2, unrolled and fine-tuned for {FT_EPOCHS} epochs")
    print(f"    squared error per test image: autoencoder {hist2[-1]:.2f}, first two principal components {pca2_err:.2f}")
    print(f"    5-nearest-neighbour accuracy of test codes against {n_ref} training codes (chance 0.10):")
    print(f"      autoencoder {acc_ae:.3f}   PCA {acc_pca:.3f}")
    print("    per digit (autoencoder / PCA):")
    print("      " + "  ".join(f"{d}: {knn_accuracy(z_train, y[:n_ref], z_test[yt == d], yt[yt == d]):.2f}/"
                               f"{knn_accuracy(p_train, y[:n_ref], p_test[yt == d], yt[yt == d]):.2f}" for d in range(5)))
    print("      " + "  ".join(f"{d}: {knn_accuracy(z_train, y[:n_ref], z_test[yt == d], yt[yt == d]):.2f}/"
                               f"{knn_accuracy(p_train, y[:n_ref], p_test[yt == d], yt[yt == d]):.2f}" for d in range(5, 10)))


if __name__ == "__main__":
    main()

