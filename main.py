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
import argparse
import numpy as np
from apex import amp
from tqdm import tqdm
from model_zoo import *
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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
    
#PGD attack
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


def mask_label(images,labels,delta,lamda1):
    #每一批batch_size区域一样
    # labels是one-hot编码
    batch,kernal,H,W = images.shape
    rx1 = int(np.random.uniform(0,W))
    rx2 = int(min(W,W*np.sqrt((1-lamda1))+rx1))
    ry1 = int(np.random.uniform(0,H))
    ry2 = int(min(H,H*np.sqrt((1-lamda1))+ry1))
    mask = torch.zeros((batch,kernal,H,W))
    mask[:,:,ry1:ry2,rx1:rx2] = 1
    mask = mask.to(device)
    P_image = images + delta*mask
    P_image_reverse = images + delta*(1-mask)
    lamda1_hat = ((rx2-rx1)*(ry2-ry1))/(H*W)
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
                delta = pgd_linf(model,inputs,targets,alpha = 2/255,num_iter = 10,randomize = True)
                adv = inputs+delta
            outputs = model(adv)
            _, predicted = outputs.max(1)
            total = total+targets.size(0)
            correct = correct+predicted.eq(targets).sum().item()
            iterator.set_description(str(predicted.eq(targets).sum().item()/targets.size(0)))


    # Save checkpoint.
    acc = 100.*correct/total
    print('test_acc:',acc)
    
    with open("./mix_wide28_cifar100.txt","a+") as f:
        f.write(str(acc)+'\n')


def train(epoch):
    model.train()
    iterator = tqdm(trainloader, ncols=0, leave=False)
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(iterator):
        inputs, targets = inputs.to(device), targets.to(device)
        onehot = torch.eye(10)[targets].to(device)
        #生成扰动
        noise = torch.FloatTensor(inputs.shape).uniform_(-epsilon, epsilon).to(device)
        x = inputs + noise
        x = x.clamp(min=0., max=1.)
        x.requires_grad_()
        
        for i in range(10):
            with torch.autograd.set_detect_anomaly(True):
                x.requires_grad_()
                out = model(x)
                loss = criterion.softlabel_ce(out, onehot)    #out 128*10 
                grads = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
                x = x.detach() + 2/255 * torch.sign(grads.detach())
                x = torch.min(torch.max(x, inputs - epsilon), inputs + epsilon)
                x = torch.clamp(x, min=0.0, max=1.0)
        #delta = (x - inputs) * 2
        delta = x - inputs
        '''
        x_av = inputs + delta
        x_av = x_av.clamp(min=0., max=1.)
        y_nat = label_smoothing(onehot, 10, 0.5)
        y_ver = label_smoothing(onehot, 10, 0.7)
        #policy = np.random.beta(1.0, 1.0)
        policy_x = torch.from_numpy(np.random.beta(1, 1, [x.size(0), 1, 1, 1])).float().to(device)
        policy_y = policy_x.view(x.size(0), -1)
        X = policy_x * inputs + (1 - policy_x) * x_av
        Y = policy_y * y_nat + (1 - policy_y) * y_ver
        '''
        
        
        #生成两种部分扰动样本和对应标签
        lamda1 = np.random.uniform()
        P_image,P_image_reverse,ti,ti_reverse = mask_label(inputs,onehot,delta,lamda1)
        lamda2_x = torch.from_numpy(np.random.beta(1, 1, [128, 1, 1, 1])).float().to(device)
        lamda2_y = lamda2_x.view(inputs.size(0), -1)
        #data mixing
        X = lamda2_x*P_image+(1-lamda2_x)*P_image_reverse
        Y = lamda2_y*ti+(1-lamda2_y)*ti_reverse
        #print(Y)
        #print(targets)
        #模型训练
        
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion.softlabel_ce(outputs, Y)
        #with amp.scale_loss(loss, optimizer) as scaled_loss:
        #    scaled_loss.backward()
        loss.backward()
        optimizer.step()
        train_loss =train_loss+loss.item()
        _, predicted = outputs.max(1)
        total =total+targets.size(0)

        correct =correct+predicted.eq(targets).sum().item()
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
    
    if not os.path.exists('./mix_wide34_cifar100'):
        os.mkdir('./mix_wide34_cifar100')
    torch.save(state, './mix_wide34_cifar100/ckpt.{}'.format(epoch))
    best_acc = acc
        
        
def adjust_learning_rate(optimizer, epoch):
    if epoch < 100:
        lr = 0.1
    elif epoch >= 100 and epoch < 150:
        lr = 0.01
    elif epoch >= 150:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # Normalization messes with l-inf bounds.
])
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True)

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = CustomLossFunction()

#model = PreActResNet18(num_classes=10,activation_fn=nn.ReLU).to(device)
model = WRN_34_10(num_classes=10, conv1_size=3, dropout=0.1, activation_fn=nn.ReLU).cuda()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
#model,optimizer = amp.initialize(model,optimizer,opt_level="O2")
epsilon = 8.0/255

start_epoch = 0
resume = None
#resume = 72
if resume is not None:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('./mix_wide28-10'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./mix_wide28-10/ckpt.{}'.format(resume))
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])



    
for epoch in range(start_epoch, 200):
    adjust_learning_rate(optimizer, epoch)
    time1 = time.time()
    train(epoch)
    if(epoch>150):
        test()
    time2 = time.time()
    print(time2-time1)
