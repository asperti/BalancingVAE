This repository contains an implementation of Variational Autoencoders with a new balancing strategy between reconstrution error and KL-divergence in the loss function.

Specifically, we enforce a constant balance between these two components via normalization of the reconstruction error by an estimation of its current value, derived from minibatches.

We derived this technique by an investigation of the loss function used by Dai e Wipf for their Two-Stage VAE, where the  balancing  parameter  was instead  learned  during training.

Our  technique  seems  to  outperform  all  previous Variational Approaches, permitting us to obtain unprecedented FID scores for traditional datasets such as CIFAR-10 and CelebA.

The code is largely based on Dai e Wipf code at
https://github.com/daib13/TwoStageVAE

The code for computing fid is a minor adaptation of the code at
https://github.com/tsc2017/Frechet-Inception-Distance