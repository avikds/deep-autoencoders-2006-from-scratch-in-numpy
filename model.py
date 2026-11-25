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

