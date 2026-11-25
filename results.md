```
[1] MNIST subset: 20000 training and 1000 test digits, 784 intensities in [0, 1]
    pixels below 0.1: 0.83, above 0.9: 0.09, in between: 0.09
    squared error per test image when every digit is 'reconstructed' as the average image: 52.55

[2] greedy pretraining of 4 RBMs, 2 passes each (the paper: 50), mini-batches of 100
    RBM 784-500 (binary hidden units, lr 0.1): reconstruction error per vector 21.19 -> 12.73
    RBM 500-250 (binary hidden units, lr 0.1): reconstruction error per vector 10.46 -> 6.09
    RBM 250-125 (binary hidden units, lr 0.1): reconstruction error per vector 8.75 -> 5.27
    RBM 125-30  (linear Gaussian code units, lr 0.001): reconstruction error per vector 16.05 -> 8.57

[3] unrolled autoencoder 784-500-250-125-30-125-250-500-784: 8 weight matrices, 1,106,564 parameters
    squared error per test image before fine-tuning: 27.30

[4] fine-tuning with back-propagation of the cross-entropy, 6 epochs, lr 0.02, momentum 0.9
    pretrained, test error after each epoch:   12.95  12.04  11.53  11.25  10.66  10.56
    random N(0, 0.1) start (226.41 before):   52.78  46.38  43.79  39.50  36.75  32.62
    the random network's reconstructions differ from image to image by 0.107 per pixel on average (the test images themselves: 0.190)

[5] squared error per test image with 30-dimensional codes
    method                         this run   paper (all 60,000 digits, 200 epochs)
    deep autoencoder, pretrained      10.56   3.00
    same net, random start            32.62   (not reported for MNIST)
    logistic PCA (784-30-784)          8.36   8.01
    PCA                               13.71   13.87
    average image, for reference      52.55
    In the paper the fully trained deep autoencoder (3.00) beats logistic PCA (8.01) and PCA (13.87).
    Here, after 2 pretraining and 6 fine-tuning epochs on 20,000 digits, it beats PCA but not
    logistic PCA. The cause is the training budget: logistic PCA has one layer of 23,520 weights on each side and
    is nearly converged in a few epochs, while the deep net's 1.1 million parameters are still improving every
    epoch (see [4]). To close the gap, raise FT_EPOCHS and RBM_EPOCHS towards the paper's 200 and 50, train on
    more digits, or use the paper's 784-1000-500-250-30 sizes, if you have the time to run it.
    the first test digit, its autoencoder reconstruction and its PCA reconstruction:
      ....................   ....................   ....................
      ..........+##+......   .........++###+.....   ........++###++.....
      ........+######.....   ........########++..   .......+#######++...
      ......+#########+...   ......+##########+..   ......+#########++..
      ......###########...   ......############..   ......+##########+..
      .....+#####....##+..   .....+####...+####..   .....+####+..+###+..
      .....+####.....##+..   .....####+...+###+..   .....+####...+###+..
      ......###+...+###...   .....+###++++####...   .....+###+...+###+..
      .....+##########....   .....+##########....   .....+####++####+...
      ......#########.....   ......#########+....   ......#########+....
      ......+#######+.....   ......+########.....   ......+########+....
      ......+########.....   ......+########+....   ......+########+....
      ......####+.+##+....   ......####++####....   .....+####+++##+....
      .....####....###....   .....####+..+###....   ....+####+..+###....
      .....###.....###....   .....###+....###....   ....+####...+###....
      .....###.....##+....   ....+###....+###....   ....+###+...+##+....
      .....##+....###.....   ....####....###+....   ....+###...+###+....
      .....###..+####.....   ....+###+.+####.....   ....+###+++####.....
      .....#########......   ....+#########......   .....+########+.....
      .....+#######.......   .....########.......   .....+#######+......

[6] 2-D codes: the same three pretrained RBMs, a new top RBM 125-2, unrolled and fine-tuned for 6 epochs
    squared error per test image: autoencoder 40.37, first two principal components 43.36
    5-nearest-neighbour accuracy of test codes against 5000 training codes (chance 0.10):
      autoencoder 0.606   PCA 0.437
    per digit (autoencoder / PCA):
      0: 0.94/0.80  1: 0.91/0.95  2: 0.57/0.25  3: 0.73/0.42  4: 0.52/0.41
      5: 0.41/0.18  6: 0.57/0.41  7: 0.56/0.46  8: 0.40/0.14  9: 0.35/0.19
```
