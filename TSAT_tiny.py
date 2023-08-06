import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import math
import os
import random
import argparse
import numpy as np
import foolbox as fb
import cv2 as cv
from tqdm import tqdm
from TinyImageNet_models import *
from TinyImageNet_utils import *
#from apex import amp # accurate with mixed precision training
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    parser = argparse.ArgumentParser('TSAT')
    parser.add_argument('--batch_size',default=128,type=int)
    parser.add_argument('--data_dir',default='../data',type=str)
    parser.add_argument('--data_type',default='Tiny-ImageNet',type=str)
    parser.add_argument('--out_dir',default='result',type=str,help='Output directory')
    parser.add_argument('--initial_lr',default=0.1,type=float,help='initial learning rate')
    parser.add_argument('--epoch_num',default=200,type=int)
    parser.add_argument('--model_type',default='ResNet18',type=str,choices=['ResNet18', 'PreActResNet18','WideResNet34'])
    parser.add_argument('--epsilon',default=8.0/255,type=float)
    parser.add_argument('--step_size',default=2.0/255,type=float)
    parser.add_argument('--Deta',default=0.05,type=float,help='parameter of LBE')
    
    arguments = parser.parse_args()
    return arguments

args = get_args()


class CustomLossFunction:
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        
    def softlabel_ce(self, x, t):
        b, c = x.shape
        x_log_softmax = torch.log_softmax(x, dim=1)
        if self.reduction == 'mean':
            loss = -torch.sum(t*x_log_softmax) / b
        elif self.reduction == 'sum':
            loss = -torch.sum(t*x_log_softmax)
        elif self.reduction == 'none':
            loss = -torch.sum(t*x_log_softmax, keepdims=True)
        return loss

def label_smoothing(onehot, n_classes, factor):
    return onehot * factor + (onehot - 1) * ((factor - 1)/(n_classes - 1))
    
def pgd_linf(model, X, y, epsilon=8 / 255, alpha=2 / 255, num_iter=20, randomize=False, conti=False, initial=None):
    if conti == True:
        delta = initial
    elif randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
        
    return delta.detach()


def adaptive_binary(img_tensor):
    Block_size = [3,5,7]
    block_size = random.sample(Block_size,1)[0]
    sobel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).to(device)
    sobel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0).to(device)
    image_r, image_g, image_b = img_tensor.split(1, dim=1)
    image_r=image_r.to(device)
    sobel_r_x = torch.nn.functional.conv2d(image_r, sobel_x, padding=1)
    sobel_r_y = torch.nn.functional.conv2d(image_r, sobel_y, padding=1)
    sobel_r = torch.sqrt(torch.pow(sobel_r_x, 2) + torch.pow(sobel_r_y, 2))
    # Calculate the average of the neighborhood around each pixel
    kernel = torch.ones(1, 1, block_size, block_size) / (block_size ** 2)
    kernel = kernel.to(device)
    # Padding is set to block_size // 2 to ensure that the output size is consistent with the input size
    mean_img = F.conv2d(sobel_r, kernel, padding=block_size // 2)
    # Binarize the image
    threshold = (mean_img - args.Deta).to(device)
    threshold = (mean_img).to(device) 
    bin_img = torch.where(sobel_r >= threshold, torch.tensor(1.).to(device), torch.tensor(0.).to(device))
    return bin_img

def mask_label(images,labels,delta,lamda1):
    batch,kernal,H,W = images.shape
    if(epoch<150):
        mask = adaptive_binary(images)
    else:
        probs = torch.rand((args.batch_size,))
        probs = probs.view((args.batch_size, 1, 1, 1)).expand((args.batch_size, 1, H, W))
        mask = torch.bernoulli(probs) #128 1 32 32

    mask = mask.to(device)
    lamda1_hat = torch.sum(mask.squeeze(1),dim = [1,2])/(H*W)
    lamda1_hat = torch.reshape(lamda1_hat,(args.batch_size,1)).repeat(1,num_classes)
    P_image = images + delta*mask
    P_image_reverse = images + delta*(1-mask)
    ti = lamda1_hat*labels+(1-lamda1_hat)*(1-labels)*(1/(labels.shape[1]-1))
    ti_reverse = lamda1_hat*(1-labels)*(1/(labels.shape[1]-1))+(1-lamda1_hat)*labels
    return P_image,P_image_reverse,ti,ti_reverse

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        iterator = tqdm(testloader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.enable_grad():
                delta = pgd_linf(model,inputs,targets,alpha = args.step_size, num_iter = 10,randomize = True)
                adv = inputs+delta
            outputs = model(adv)
            _, predicted = outputs.max(1)
            total =total + targets.size(0)
            correct =correct + predicted.eq(targets).sum().item()
            iterator.set_description(str(predicted.eq(targets).sum().item()/targets.size(0)))


    # Save checkpoint.
    acc = 100.*correct/total
    print('test_acc:',acc)
    
    with open(output_file,"a+") as f:
        f.write(str(acc)+'\n')


def train(epoch):
    model.train()
    iterator = tqdm(trainloader, ncols=0, leave=False)
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(iterator):
        inputs, targets = inputs.to(device), targets.to(device)
        onehot = torch.eye(num_classes)[targets].to(device)
        #生成扰动
        noise = torch.FloatTensor(inputs.shape).uniform_(-args.epsilon, args.epsilon).to(device)
        x = inputs + noise
        x = x.clamp(min=0., max=1.)
        x.requires_grad_()
        
        for i in range(10):
            x.requires_grad_()
            out = model(x)
            loss = criterion.softlabel_ce(out, onehot)    
            grads = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
            
            x = x.detach() + args.step_size * torch.sign(grads.detach())
            x = torch.min(torch.max(x, inputs - args.epsilon), inputs + args.epsilon)
            x = torch.clamp(x, min=0.0, max=1.0)
        delta = x - inputs

        #Generate two kinds of partially perturbed samples and corresponding labels
        lamda1 = np.random.uniform()
        P_image,P_image_reverse,ti,ti_reverse = mask_label(inputs,onehot,delta,lamda1)
        lamda2_x = torch.from_numpy(np.random.beta(1, 1, [args.batch_size, 1, 1, 1])).float().to(device)
        lamda2_y = lamda2_x.view(inputs.size(0), -1)
        #data mixing
        X = lamda2_x*P_image+(1-lamda2_x)*P_image_reverse
        Y = lamda2_y*ti+(1-lamda2_y)*ti_reverse
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion.softlabel_ce(outputs, Y)
        #with amp.scale_loss(loss, optimizer) as scaled_loss:
        #    scaled_loss.backward()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss =train_loss + loss.item()
        _, predicted = outputs.max(1)
        total =total + targets.size(0)

        correct =correct + predicted.eq(targets).sum().item()
        iterator.set_description(str(predicted.eq(targets).sum().item() / targets.size(0)))

    acc = 100.*correct/total
    print('Train acc:', acc)


    print('Saving..')
    state = {
        'model': model.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }        
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    torch.save(state, output_dir+'/ckpt.{}'.format(epoch))
    best_acc = acc
        
        
num_classes = 200
output_dir = os.path.join(args.out_dir,'model_'+args.model_type+'_'+args.data_type)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_file = os.path.join(output_dir, 'output.log')

trainloader, testloader =  New_ImageNet_get_loaders_64(args.data_dir, args.batch_size)


if args.model_type == 'ResNet18':
    model = ResNet18().to(device)
elif args.model_type == 'PreActResNet18':
    model = PreActResNet18().to(device)
else:
    model = WideResNet().to(device)


criterion = CustomLossFunction()
optimizer = optim.SGD([{'params': model.parameters(), 'initial_lr': 0.1}], 0.1, momentum=0.9, weight_decay=5e-4)
lr_steps = args.epoch_num * len(trainloader)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0,max_lr=0.1,step_size_up=lr_steps / 2,step_size_down=lr_steps / 2)
#model,optimizer = amp.initialize(model,optimizer,opt_level="O2")
start_epoch = 0
resume = None

if resume is not None:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('./mix_sobel_avgsum_wideres'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./mix_sobel_avgsum_wideres/ckpt.{}'.format(resume))
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])
 
for epoch in range(start_epoch, args.epoch_num):
    time1 = time.time()
    train(epoch)
    test()
    time2 = time.time()
    print(time2-time1)
