# Project Summary

This project investigates sparse federated learning for image classification on CIFAR-100.

The original university project compared centralized training, standard Federated Averaging and sparse Federated Averaging using a pre-trained DINO ViT-S/16 backbone. The experiments simulated both IID and non-IID client data partitions to study the impact of data heterogeneity in federated learning.

A specific contribution of the project was the implementation of a simple secure aggregation mechanism based on zero-sum noise masking. In this protocol, client updates are masked before aggregation, while the noise cancels out at the server level so that the final averaged model remains unchanged.

This public repository is a cleaned portfolio version of the project. It focuses on code structure, reusable modules and lightweight demonstrations rather than storing large datasets, checkpoints or raw experiment outputs.
