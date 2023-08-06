# TSAT
Code for ACM MM2023 "Adversarial Training of Deep Neural Networks Guided by Texture and Structural Information"

## Introduction
Adversarial training (AT) is one of the most effective ways for deep neural network models to resist adversarial examples. However, there is still a significant gap between robust training accuracy and testing accuracy. Although recent studies have shown that data augmentation can effectively reduce this gap, most methods heavily rely on generating large amounts of training data without considering which features are beneficial for model robustness, making them inefficient. To address the above issue, we propose a two-stage AT algorithm for image data that adopts different data augmentation strategies during the training process to improve model robustness. In the first stage, we focus on the convergence of the algorithm, which uses structure and texture information to guide AT. In the second stage, we introduce a strategy that randomly fuses the data features to generate diverse adversarial examples for AT. We compare our proposed algorithm with five state-of-the-art algorithms on three models, and the experimental results achieve the best robust accuracy under all evaluation metrics on the CIFAR10 dataset, demonstrating the superiority of our method.

## Requirements
Python3、Pytorch1.81、numpy
