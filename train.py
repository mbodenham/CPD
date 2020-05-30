import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import numpy as np
import pdb, os, argparse
from datetime import datetime

from gts_folder import ImageGroundTruthFolder
from model.CPD_models import CPD_VGG
from model.CPD_ResNet_models import CPD_ResNet
from utils import clip_gradient, adjust_lr


parser = argparse.ArgumentParser()
parser.add_argument('--datasets_path', type=str, default='./datasets', help='path to datasets')
parser.add_argument('--cuda', type=bool, default=True, help='run with cuda')
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
opt = parser.parse_args()

print('Learning Rate: {} ResNet: {}'.format(opt.lr, opt.is_ResNet))
# build models
if opt.is_ResNet:
    model = CPD_ResNet()
else:
    model = CPD_VGG()

if opt.cuda:
    model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

transform = transforms.Compose([
            transforms.Resize((opt.trainsize, opt.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

gt_transform = transforms.Compose([
            transforms.Resize((opt.trainsize, opt.trainsize)),
            transforms.ToTensor()])

dataset = ImageGroundTruthFolder(opt.datasets_path, transform=transform, target_transform=gt_transform)
train_loader = DataLoader(dataset, batch_size=opt.batchsize, shuffle=True)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()


def train(train_loader, model, optimizer, epoch):
    model.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        if opt.cuda:
            images = images.cuda()
            gts = gts.cuda()

        atts, dets = model(images)
        loss1 = CE(atts, gts)
        loss2 = CE(dets, gts)
        loss = loss1 + loss2
        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 400 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss2.data))

    if opt.is_ResNet:
        save_path = 'models/CPD_Resnet/'
    else:
        save_path = 'models/CPD_VGG/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), save_path + 'CPD.pth' + '.%d' % epoch)

print("Let's go!")
for epoch in range(1, opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
