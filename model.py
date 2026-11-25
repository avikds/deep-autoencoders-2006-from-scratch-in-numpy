"""
Deep Autoencoders (2006) from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - rbm_energy
import numpy as np

def sigmoid(x):
    """1 / (1 + exp(-x)) elementwise, without overflow or underflow."""

    x = np.asarray(x, dtype=float)

    # Flatten the input to handle arrays of any shape consistently.
    flat_x = x.reshape(-1)
    result = np.full(flat_x.shape, np.nan, dtype=float)

    # Determine the safe exponential calculation threshold.
    cutoff = -np.log(np.finfo(float).tiny)

    # Handle non-negative inputs within the safe exponential range.
    positive_mask = (flat_x >= 0.0) & (flat_x <= cutoff)
    result[positive_mask] = (
        1.0 / (1.0 + np.exp(-flat_x[positive_mask]))
    )

    # Very large positive inputs produce a sigmoid value of 1.
    result[flat_x > cutoff] = 1.0

    # Handle negative inputs within the safe exponential range.
    negative_mask = (flat_x < 0.0) & (flat_x >= -cutoff)
    exp_x = np.exp(flat_x[negative_mask])

    result[negative_mask] = exp_x / (1.0 + exp_x)

    # Very large negative inputs produce a sigmoid value of 0.
    result[flat_x < -cutoff] = 0.0

    # Restore the original input shape.
    result = result.reshape(x.shape)

    # Return a scalar for scalar input.
    return result.item() if result.ndim == 0 else result


def rbm_energy(v, h, W, b_vis, b_hid):
    """Energy of each row: -v.b_vis - h.b_hid - v W h. Returns shape (n,)."""

    # Visible-unit bias contribution.
    visible_term = np.sum(v * b_vis, axis=1)

    # Hidden-unit bias contribution.
    hidden_term = np.sum(h * b_hid, axis=1)

    # Visible-hidden interaction contribution.
    interaction_term = np.sum((v @ W) * h, axis=1)

    # Calculate the energy for each sample.
    return -(visible_term + hidden_term + interaction_term)

# Step 2 - cd1_gradients
def cd1_gradients(v0, W, b_vis, b_hid, rng):
    """One CD-1 step: sampled binary hidden states, real-valued confabulation.
    
    Returns (dW, db_vis, db_hid, v1).
    """

    # Step 1: Compute hidden-unit activation probabilities from the data.
    p0 = sigmoid(v0 @ W + b_hid)

    # Step 2: Sample binary hidden states.
    # This is the only random sampling operation in CD-1.
    h0 = (rng.random(p0.shape) < p0).astype(float)

    # Step 3: Compute visible-unit probabilities from the sampled hidden states.
    # Keep the probabilities as real-valued values; do not sample binary visibles.
    v1 = sigmoid(h0 @ W.T + b_vis)

    # Step 4: Compute hidden-unit probabilities from the reconstructed visibles.
    # These probabilities are also kept as real-valued values.
    p1 = sigmoid(v1 @ W + b_hid)

    # Number of samples in the mini-batch.
    n = v0.shape[0]

    # Step 5: Calculate the positive-phase minus negative-phase weight gradient.
    dW = (v0.T @ p0 - v1.T @ p1) / n

    # Calculate the visible-unit bias gradient.
    db_vis = np.mean(v0 - v1, axis=0)

    # Calculate the hidden-unit bias gradient.
    db_hid = np.mean(p0 - p1, axis=0)

    return dW, db_vis, db_hid, v1

# Step 3 - cd1_gradients_linear
def cd1_gradients_linear(v0, W, b_vis, b_hid, rng):
    """CD-1 with linear Gaussian hidden units (unit variance).
    
    Returns (dW, db_vis, db_hid, v1).
    """

    # Step 1: Calculate the mean activations of the hidden code units
    # using the original visible data.
    m0 = v0 @ W + b_hid

    # Step 2: Sample real-valued hidden states by adding Gaussian noise
    # with zero mean and unit variance.
    # This is the only random sampling operation in this step.
    h0 = m0 + rng.standard_normal(m0.shape)

    # Step 3: Reconstruct the visible-unit probabilities.
    # The visible units remain logistic, so we apply the sigmoid function.
    # Do not sample binary visible states.
    v1 = sigmoid(h0 @ W.T + b_vis)

    # Step 4: Calculate the hidden-unit means from the reconstructed data.
    # The hidden units are linear, so no sigmoid is applied.
    m1 = v1 @ W + b_hid

    # Number of samples in the mini-batch.
    n = v0.shape[0]

    # Step 5: Calculate the weight gradient using positive-phase
    # and negative-phase statistics.
    dW = (v0.T @ m0 - v1.T @ m1) / n

    # Calculate the visible-unit bias gradient.
    db_vis = np.mean(v0 - v1, axis=0)

    # Calculate the hidden-unit bias gradient.
    db_hid = np.mean(m0 - m1, axis=0)

    return dW, db_vis, db_hid, v1

# Step 4 - train_rbm
def train_rbm(data, n_hidden, epochs, rng, linear_hidden=False, lr=0.1, momentum=0.9, weight_cost=0.0002, batch_size=100):
    """Mini-batch CD-1 with momentum and a weight penalty.
    
    Returns (W, b_vis, b_hid, errors per epoch).
    """

    # Get the number of training samples and visible units.
    n, n_vis = data.shape

    # Initialize weights first using the specified random generator.
    W = 0.1 * rng.standard_normal((n_vis, n_hidden))

    # Initialize visible and hidden biases to zero.
    b_vis = np.zeros(n_vis)
    b_hid = np.zeros(n_hidden)

    # Initialize momentum increments for weights and biases.
    dW_inc = np.zeros_like(W)
    dbv_inc = np.zeros_like(b_vis)
    dbh_inc = np.zeros_like(b_hid)

    # Store the reconstruction error after each epoch.
    errors = []

    # Determine the number of mini-batches.
    # Any leftover samples are added to the final mini-batch.
    n_batches = max(1, n // batch_size)

    # Train the RBM for the specified number of epochs.
    for epoch in range(epochs):

        # Randomly shuffle the training sample indices.
        order = rng.permutation(n)

        # Accumulate squared reconstruction errors for this epoch.
        epoch_error = 0.0

        # Process the mini-batches in shuffled order.
        for batch_idx in range(n_batches):

            # Calculate the mini-batch boundaries.
            start = batch_idx * batch_size

            # The final batch takes all remaining samples.
            end = n if batch_idx == n_batches - 1 else start + batch_size

            # Extract the indices for the current mini-batch.
            batch = order[start:end]

            # Get the original visible-unit data.
            v0 = data[batch]

            # Perform one CD-1 step using the appropriate hidden-unit model.
            if linear_hidden:
                dW, db_vis, db_hid, v1 = cd1_gradients_linear(
                    v0, W, b_vis, b_hid, rng
                )
            else:
                dW, db_vis, db_hid, v1 = cd1_gradients(
                    v0, W, b_vis, b_hid, rng
                )

            # Update the weight momentum increment.
            # Apply the weight penalty only to the weights.
            dW_inc = momentum * dW_inc + lr * (dW - weight_cost * W)

            # Update the weights.
            W += dW_inc

            # Update the visible-bias momentum increment.
            dbv_inc = momentum * dbv_inc + lr * db_vis

            # Update the visible biases.
            b_vis += dbv_inc

            # Update the hidden-bias momentum increment.
            dbh_inc = momentum * dbh_inc + lr * db_hid

            # Update the hidden biases.
            b_hid += dbh_inc

            # Accumulate squared reconstruction error using
            # the visible reconstruction returned by the CD-1 step.
            epoch_error += np.sum((v0 - v1) ** 2)

        # Calculate the average reconstruction error per sample.
        errors.append(float(epoch_error / n))

    return W, b_vis, b_hid, errors

# Step 5 - pretrain_stack
def pretrain_stack(data, sizes, epochs, rng, lr=0.1, top_lr=0.001):
    """Greedy RBM stack; the top RBM has linear Gaussian hidden units.
    
    Returns (stack, errors).
    """

    # Store the RBM parameters and training errors in order.
    stack = []
    errors = []

    # The number of RBMs is one less than the number of layer sizes.
    n_rbms = len(sizes) - 1

    # Start pretraining with the original input data.
    layer_input = data

    # Train the RBMs sequentially, starting from the bottom layer.
    for k in range(n_rbms):

        # Identify whether the current RBM is the top-level RBM.
        is_top = (k == n_rbms - 1)

        # Select the appropriate learning rate and hidden-unit type.
        current_lr = top_lr if is_top else lr
        linear_hidden = is_top

        # Train the current RBM on the input received from the previous layer.
        W, b_vis, b_hid, rbm_errors = train_rbm(
            layer_input,
            sizes[k + 1],
            epochs,
            rng,
            linear_hidden=linear_hidden,
            lr=current_lr
        )

        # Save the trained RBM parameters (bottom RBM first).
        stack.append((W, b_vis, b_hid))

        # Save the reconstruction errors for this RBM.
        errors.append(rbm_errors)

        # For every RBM except the top one, calculate hidden probabilities.
        # These probabilities become the input to the next RBM.
        if not is_top:
            layer_input = sigmoid(layer_input @ W + b_hid)

    return stack, errors

# Step 6 - unroll
def unroll(stack):
    """Encoder (W, b_hid) bottom-up, then decoder (W.T, b_vis) top-down; all copies."""

    layers = []

    # Build the encoder by traversing the RBM stack from bottom to top.
    for W, b_vis, b_hid in stack:

        # Copy the weights and hidden biases to prevent shared memory.
        encoder_W = W.copy()
        encoder_b = b_hid.copy()

        layers.append((encoder_W, encoder_b))

    # Build the decoder by traversing the RBM stack in reverse order.
    for W, b_vis, b_hid in reversed(stack):

        # Transpose and copy the weights for the decoder.
        decoder_W = W.T.copy()

        # Copy the visible biases for the decoder layer.
        decoder_b = b_vis.copy()

        layers.append((decoder_W, decoder_b))

    return layers

# Step 7 - autoencoder_forward
def autoencoder_forward(layers, x):
    """All activations [x, a1, ..., reconstruction]; the code layer (index K - 1) is linear, the rest logistic."""

    # The total number of layers is twice the number of encoder layers.
    K = len(layers) // 2

    # Store all activations, starting with the original input.
    activations = [x]

    # Perform the forward pass through every encoder and decoder layer.
    for i, (W, b) in enumerate(layers):

        # Calculate the linear transformation for the current layer.
        s = activations[-1] @ W + b

        # The code layer (last encoder layer) uses a linear activation.
        if i == K - 1:
            a_next = s

        # All other layers, including the decoder and output layer,
        # use the logistic sigmoid activation.
        else:
            a_next = sigmoid(s)

        # Store the activation for use by the next layer.
        activations.append(a_next)

    return activations

# Step 8 - reconstruction_errors
def reconstruction_errors(x_hat, x):
    """(cross-entropy, squared error), each summed over pixels and averaged over images, as Python floats."""

    # Clip reconstructed pixel probabilities to avoid log(0).
    # This ensures numerically stable cross-entropy calculations.
    x_hat_clipped = np.clip(x_hat, 1e-10, 1.0 - 1e-10)

    # Calculate the binary cross-entropy for each pixel.
    cross_entropy_per_pixel = (
        x * np.log(x_hat_clipped)
        + (1.0 - x) * np.log(1.0 - x_hat_clipped)
    )

    # Sum cross-entropy over pixels for each image,
    # then average across the mini-batch.
    cross_entropy = -np.mean(
        np.sum(cross_entropy_per_pixel, axis=1)
    )

    # Calculate squared differences between original and reconstructed data.
    squared_differences = (x - x_hat) ** 2

    # Sum squared errors over pixels for each image,
    # then average across the mini-batch.
    squared_error = np.mean(
        np.sum(squared_differences, axis=1)
    )

    # Return both metrics as standard Python floats.
    return float(cross_entropy), float(squared_error)

# Step 9 - autoencoder_backward
def autoencoder_backward(layers, acts, x):
    """Gradients [(dW, db), ...] of the mean cross-entropy, one per layer, by back-propagation."""

    # Number of encoder layers.
    K = len(layers) // 2

    # Number of training samples in the batch.
    n = x.shape[0]

    # Initialize the output-layer gradient.
    # For sigmoid output units with cross-entropy loss,
    # the derivative with respect to the output pre-activation
    # simplifies to (reconstruction - original_input) / n.
    d = (acts[-1] - x) / n

    # Store gradients in the original layer order.
    grads = [None] * len(layers)

    # Back-propagate from the output layer to the input layer.
    for i in range(len(layers) - 1, -1, -1):

        # Retrieve the current layer's weights and biases.
        W, b = layers[i]

        # Calculate the weight gradient.
        dW = acts[i].T @ d

        # Calculate the bias gradient.
        db = np.sum(d, axis=0)

        # Store the gradients at the corresponding layer index.
        grads[i] = (dW, db)

        # Propagate the gradient to the preceding layer.
        if i > 0:

            # Gradient with respect to the preceding layer's activation.
            d = d @ W.T

            # Apply the sigmoid derivative if the preceding layer
            # is not the linear code layer.
            if i - 1 != K - 1:

                a = acts[i]
                d = d * a * (1.0 - a)

            # The code layer is linear, so its derivative is 1.
            # Therefore, no activation derivative is applied there.

    return grads

# Step 10 - random_autoencoder
def random_autoencoder(sizes, rng, std=0.1):
    """Mirrored autoencoder with N(0, std^2) weights (one draw per layer, input to output) and zero biases."""

    # Construct the complete mirrored architecture.
    # The decoder mirrors the encoder, excluding the code layer.
    layer_sizes = list(sizes) + list(sizes[-2::-1])

    # Store the independently initialized weights and biases.
    layers = []

    # Initialize every layer sequentially from input to output.
    for i in range(len(layer_sizes) - 1):

        # Determine the input and output dimensions of the current layer.
        n_in = layer_sizes[i]
        n_out = layer_sizes[i + 1]

        # Initialize weights using a Gaussian distribution.
        # Each layer receives its own independent random draw.
        W = std * rng.standard_normal((n_in, n_out))

        # Initialize biases to zero.
        b = np.zeros(n_out)

        # Store the weight matrix and bias vector.
        layers.append((W, b))

    return layers

# Step 11 - finetune
def finetune(layers, x, x_eval, epochs, rng, lr=0.02, momentum=0.9, batch_size=100):
    """Mini-batch gradient descent with momentum on the cross-entropy, layers updated in place.
    
    Returns the squared error per image on x_eval after each epoch.
    """

    # Initialize one velocity matrix for each weight matrix
    # and one velocity vector for each bias vector.
    velocities = [
        (np.zeros_like(W), np.zeros_like(b))
        for W, b in layers
    ]

    # Store the evaluation reconstruction error after each epoch.
    history = []

    # Number of training samples.
    n = len(x)

    # Train the autoencoder for the specified number of epochs.
    for epoch in range(epochs):

        # Randomly shuffle the training sample indices.
        order = rng.permutation(n)

        # Process the shuffled data in mini-batches.
        # The final batch may contain fewer than batch_size samples.
        for start in range(0, n, batch_size):

            # Select the indices for the current mini-batch.
            batch = order[start:start + batch_size]

            # Extract the current mini-batch.
            x_batch = x[batch]

            # Perform the forward pass through the autoencoder.
            acts = autoencoder_forward(layers, x_batch)

            # Calculate the gradients using backpropagation.
            grads = autoencoder_backward(layers, acts, x_batch)

            # Update every layer using gradient descent with momentum.
            for i in range(len(layers)):

                # Retrieve the current weights and biases.
                W, b = layers[i]

                # Retrieve the corresponding gradients.
                dW, db = grads[i]

                # Retrieve the previous momentum velocities.
                vW, vb = velocities[i]

                # Update the momentum velocities.
                vW = momentum * vW - lr * dW
                vb = momentum * vb - lr * db

                # Update the weights and biases.
                W = W + vW
                b = b + vb

                # Store the updated velocities.
                velocities[i] = (vW, vb)

                # Replace the layer parameters in place.
                layers[i] = (W, b)

        # Evaluate the autoencoder after completing the current epoch.
        eval_acts = autoencoder_forward(layers, x_eval)

        # Calculate the squared reconstruction error per image.
        _, squared_error = reconstruction_errors(
            eval_acts[-1], x_eval
        )

        # Store one Python float per epoch.
        history.append(float(squared_error))

    return history

# Step 12 - fit_pca
def fit_pca(x, k):
    """(mean, components): the k leading principal directions as rows,
    orthonormal, largest variance first.
    """

    # Calculate the mean of each feature across all samples.
    mean = np.mean(x, axis=0)

    # Center the data by subtracting the feature-wise mean.
    x_centered = x - mean

    # Perform Singular Value Decomposition on the centered data.
    # Vt contains the principal directions as rows, ordered by
    # decreasing singular value (and therefore decreasing variance).
    _, _, Vt = np.linalg.svd(x_centered, full_matrices=False)

    # Select the first k principal components.
    # Each component is a row with shape (d,).
    components = Vt[:k].copy()

    return mean, components


def pca_reconstruct(x, mean, components):
    """Reconstruct each row from its k principal coordinates."""

    # Center the input data using the training-set mean.
    x_centered = x - mean

    # Project the centered data onto the principal components.
    # These are the low-dimensional PCA coordinates.
    codes = x_centered @ components.T

    # Project the coordinates back into the original feature space.
    x_reconstructed = codes @ components

    # Restore the original feature-wise mean.
    x_reconstructed += mean

    return x_reconstructed

# Step 13 - knn_accuracy
def knn_accuracy(train_codes, train_labels, test_codes, test_labels, k=5):
    """Fraction of test codes whose k nearest training codes vote for the right label."""

    # Store the predicted labels for all test samples.
    predictions = []

    # Process each test code individually.
    for test_code in test_codes:

        # Calculate Euclidean distances between the test code
        # and every training code.
        distances = np.linalg.norm(
            train_codes - test_code,
            axis=1
        )

        # Sort training indices by distance.
        # Stable sorting ensures that earlier training samples
        # win when two or more distances are equal.
        nearest_indices = np.argsort(
            distances,
            kind="stable"
        )[:k]

        # Retrieve the labels of the k nearest training samples.
        nearest_labels = train_labels[nearest_indices]

        # Count the occurrences of each digit class.
        # argmax selects the smallest label when vote counts tie.
        predicted_label = np.bincount(
            nearest_labels,
            minlength=10
        ).argmax()

        # Store the predicted digit label.
        predictions.append(predicted_label)

    # Convert predictions into a NumPy array.
    predictions = np.asarray(predictions)

    # Calculate classification accuracy.
    accuracy = np.mean(predictions == test_labels)

    # Return the accuracy as a standard Python float.
    return float(accuracy)

