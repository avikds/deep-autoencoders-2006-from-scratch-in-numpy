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

