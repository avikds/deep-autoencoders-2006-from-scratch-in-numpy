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

