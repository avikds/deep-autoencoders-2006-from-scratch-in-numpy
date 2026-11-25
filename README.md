# Deep Autoencoders (2006) from Scratch in NumPy

Reducing the Dimensionality of Data with Neural Networks (Hinton and Salakhutdinov, 2006), rebuilt in NumPy with every gradient written by hand. A deep autoencoder squeezes each 784-pixel MNIST digit through a small layer of 30 numbers and rebuilds the image from them. The paper's point is that such a network only trains well when its weights start near a good solution, and it shows how to find that start. Build its pretraining: restricted Boltzmann machines trained one layer at a time with one step of contrastive divergence, each learning features of the previous layer's feature probabilities, and a top machine whose 30 code units are linear with Gaussian noise. Unroll the stack into an encoder and a decoder that start from the same weights, transposed, and fine-tune the whole network with back-propagation of the cross-entropy reconstruction error. Then run the paper's comparison on 20,000 real digits: the pretrained autoencoder against the same network started from small random weights, which learns far more slowly and ends with about three times the error, and against logistic PCA and PCA with 30 components. In this short run the deep network beats PCA but not yet logistic PCA, which the paper's fully trained network does; the script says why and what to change. Finish with two-dimensional codes, where the autoencoder keeps the ten digit classes clearly better apart than the first two principal components.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** rbm_energy
- [x] **2.** cd1_gradients

---

Built on Deep-ML.
