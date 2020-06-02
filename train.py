import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import os, argparse
from datetime import datetime

from gts_folder import ImageGroundTruthFolder
from model.CPD_models import CPD_VGG, CPD_VGG_attention
from model.CPD_ResNet_models import CPD_ResNet
from utils import clip_gradient, adjust_lr

parser = argparse.ArgumentParser()
parser.add_argument('--datasets_path', default='./datasets/train', help='path to datasets')
parser.add_argument('--cuda', default='cuda', help='run with cuda')
parser.add_argument('--attention', action='store_true', help='attention branch model')
parser.add_argument('--resnet', action='store_true', help='VGG or ResNet backbone')
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=10, help='training batch size')
parser.add_argument('--imgres', type=int, default=352, help='dataset image training resolution')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
opt = parser.parse_args()

def train(train_loader, model, optimizer, epoch):
    print('Started training')
    total_step = len(train_loader)
    CE = torch.nn.BCEWithLogitsLoss()
    model.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        if opt.cuda == 'cuda':
            images = images.cuda()
            gts = gts.cuda()
        if opt.attention:
            atts = model(images)
            loss = CE(atts, gts)
            loss.backward()
        else:
            atts, dets = model(images)
            loss1 = CE(atts, gts)
            loss2 = CE(dets, gts)
            loss = loss1 + loss2
            loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 100 == 0 or i == total_step:
            if opt.attention:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))
            else:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Attention loss: {:.4f}, Detection loss: {:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss2.data))

    save_path = 'models/{}/'.format(model.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), '{}{}.pth.{%03d}'.format(save_path, mode.name, epoch))

print('CUDA GPU available: {}'.format(torch.cuda.is_available()))
# build models
if opt.resnet:
    model = CPD_ResNet()
elif opt.attention:
    model = CPD_VGG_attention()
else:
    model = CPD_VGG()
print('Loaded', model.name)

if opt.cuda == 'cuda':
    model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

transform = transforms.Compose([
            transforms.Resize((opt.imgres, opt.imgres)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
gt_transform = transforms.Compose([
            transforms.Resize((opt.imgres, opt.imgres)),
            transforms.ToTensor()])

dataset = ImageGroundTruthFolder(opt.datasets_path, transform=transform, target_transform=gt_transform)
train_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
print('Dataset loaded successfully')
for epoch in range(1, opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
