# MolGAN TF2+

Simplified TensorFlow 2.0+ implementation of MolGAN: An implicit generative model for small molecular graphs (https://arxiv.org/abs/1805.11973)
The original tensorflow code can be found here: https://github.com/nicola-decao/MolGAN

The current version is limited to small molecules, when it was tested on ZINC dataset of 200 000 compounds with MW 300-400 it resulted in NaN losses.

The main file is [MolGAN file](https://github.com/MiloszGrabski/MolGAN-TF2-/blob/main/MolGAN.ipynb)


![Example generated compounds](https://github.com/MiloszGrabski/MolGAN-TF2-/blob/main/MolGAN/data/example.png)
