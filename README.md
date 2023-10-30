# TSAT
Code for ACM MM2023 "Adversarial Training of Deep Neural Networks Guided by Texture and Structural Information"

## Introduction
Adversarial training (AT) is one of the most effective ways for deep neural network models to resist adversarial examples. However, there is still a significant gap between robust training accuracy and testing accuracy. Although recent studies have shown that data augmentation can effectively reduce this gap, most methods heavily rely on generating large amounts of training data without considering which features are beneficial for model robustness, making them inefficient. To address the above issue, we propose a two-stage AT algorithm for image data that adopts different data augmentation strategies during the training process to improve model robustness. In the first stage, we focus on the convergence of the algorithm, which uses structure and texture information to guide AT. In the second stage, we introduce a strategy that randomly fuses the data features to generate diverse adversarial examples for AT. We compare our proposed algorithm with five state-of-the-art algorithms on three models, and the experimental results achieve the best robust accuracy under all evaluation metrics on the CIFAR10 dataset, demonstrating the superiority of our method.

![image](https://github.com/rrr3987/examples/blob/master/1.png)

## Requirements
Python3、Pytorch1.8.1、numpy

## Train for TSAT
- On CIFAR10

  ``
  python TSAT.py --batch_size 128 --data_type cifar10 --model_type PreActResNet18 
  ``

- On CIFAR100

  ``
  python TSAT.py --batch_size 128 --data_type cifar100 --model_type PreActResNet18 
  ``

- On Tiny-ImageNet
 
  ``
  python TSAT_tiny.py --batch_size 128 --model_type PreActResNet18 --epoch_num 110
  ``
## Test model
- On CIFAR

  ``
  python evaluate.py --data_type cifar10 --model_path ./result/model_PreActResNet18_cifar10/ckpt.0 --model_type PreActResNet18
  ``

- On Tiny-ImageNet

  ``
  python evaluate_tiny.py --model_path result/model_ResNet18_cifar10 --model ResNet18
  ``
