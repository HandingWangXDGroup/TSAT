import torch
import torchvision
import numpy as np
import argparse
from advertorch.attacks import LinfPGDAttack, CarliniWagnerL2Attack
from autoattack import AutoAttack
#from model_zoo import *
import os
from TinyImageNet_utils import *
from TinyImageNet_models import *

#device_ids = [0,2,3]
#torch.cuda.set_device(device_ids[0])
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# 设置参数
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./TRADES_tiny_preact/model_preact.13', help='model path')
parser.add_argument('--model', type=str, default='WideResNet', help='model path')
parser.add_argument('--batch_size', type=int, default=200, help='batch size')
args = parser.parse_args()

def test(model, loader, attack=None):
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.cuda(), target.cuda()
        if attack is not None:
            data = attack.perturb(data, target)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    return 100.0 * correct / total

model = PreActResNet18().cuda()
#model = nn.DataParallel(model,device_ids=device_ids)
model = model.to(device)
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['model'])
Batch = 128


trainloader, testloader = New_ImageNet_get_loaders_64('../../data/tiny-imagenet-200', Batch)


'''
models_directory = './TRADES_tiny_preact'
model_files=os.listdir(models_directory)
model_files = sorted(model_files, key = lambda x: int(x.split('.')[-1]))
for model_file in model_files:
    model_path = os.path.join(models_directory,model_file)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    print(model_path)
    model.eval()
    acc_pgd_5 = test(model, testloader, pgd_attack_5)
    print(f"PGD-5 accuracy: {acc_pgd_5:.2f}%")
'''




#testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=torchvision.transforms.ToTensor())
#testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# 定义攻击模型
FGSM = LinfPGDAttack(model, loss_fn=torch.nn.CrossEntropyLoss(), eps=8/255, nb_iter=1, eps_iter=8/255,
                           rand_init=True, clip_min=0.0, clip_max=1.0)

pgd_attack_10 = LinfPGDAttack(model, loss_fn=torch.nn.CrossEntropyLoss(), eps=8/255, nb_iter=10, eps_iter=2/255,
                           rand_init=True, clip_min=0.0, clip_max=1.0)
pgd_attack_20 = LinfPGDAttack(model, loss_fn=torch.nn.CrossEntropyLoss(), eps=8/255, nb_iter=20, eps_iter=2/255,
                           rand_init=True, clip_min=0.0, clip_max=1.0)
pgd_attack_50 = LinfPGDAttack(model, loss_fn=torch.nn.CrossEntropyLoss(), eps=8/255, nb_iter=50, eps_iter=2/255,
                           rand_init=True, clip_min=0.0, clip_max=1.0)
cw_attack = CarliniWagnerL2Attack(model, num_classes=200, max_iterations=20, confidence=0, learning_rate=0.01,
                                  binary_search_steps=10, initial_const=0.001, clip_min=0.0, clip_max=1.0)
#aa_attack = AutoAttack(model, norm='Linf', eps=8/255, version='standard', seed=0)

# 测试模型


# 在测试集上进行评估



#acc_clean = test(model, testloader)
#print(f"Clean accuracy: {acc_clean:.2f}%")
#acc_fgsm = test(model, testloader, FGSM)
#print(f"FGSM accuracy: {acc_fgsm:.2f}%")
#acc_pgd_10 = test(model, testloader, pgd_attack_10)
#print(f"PGD-10 accuracy: {acc_pgd_10:.2f}%")
#acc_pgd_20 = test(model, testloader, pgd_attack_20)
#print(f"PGD-20 accuracy: {acc_pgd_20:.2f}%")
#acc_pgd_50 = test(model, testloader, pgd_attack_50)
#print(f"PGD-50 accuracy: {acc_pgd_50:.2f}%")

acc_cw = test(model, testloader, cw_attack)
print(f"CW accuracy: {acc_cw:.2f}%")


#acc_aa = test(model, testloader, aa_attack)


# 打印准确率



#
#
#print(f"PGD-50 accuracy: {acc_pgd_50:.2f}%")


#print(f"AutoAttack accuracy: {acc_aa:.2f}%")

'''
with open(f'{args.model_path}.txt', 'a') as f:
    f.write(f"Clean accuracy: {acc_clean:.2f}%\n")
    f.write(f"FGSM accuracy: {acc_fgsm:.2f}%\n")
    f.write(f"PGD-10 accuracy: {acc_pgd_10:.2f}%\n")
    f.write(f"PGD-20 accuracy: {acc_pgd_20:.2f}%\n")
    f.write(f"PGD-50 accuracy: {acc_pgd_50:.2f}%\n")
    f.write(f"CW accuracy: {acc_cw:.2f}%\n")
    #f.write(f"AutoAttack accuracy: {acc_aa:.2f}%\n")
'''
