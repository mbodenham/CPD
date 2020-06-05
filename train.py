import torch
import torchvision.transforms as transforms
import torch.utils.tensorboard  as tensorboard

import os, argparse
from datetime import datetime

from model.models import CPD, CPD_A
from model.dataset import ImageGroundTruthFolder

parser = argparse.ArgumentParser()
parser.add_argument('--datasets_path', default='./datasets/train', help='path to datasets, default = ./datasets/train')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='use cuda or cpu, default = cuda')
parser.add_argument('--attention', action='store_true', help='use attention branch model')
parser.add_argument('--imgres', type=int, default=352, help='image input and output resolution, default = 352')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs,  default = 100')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate,  default = 0.0001')
parser.add_argument('--batch_size', type=int, default=10, help='training batch size,  default = 10')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin, default = 0.5')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate, default = 0.1')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate,  default = 50')
args = parser.parse_args()

def train(train_loader, model, optimizer, epoch, writer):
    def add_image(imgs, gts, preds, step, writer):
        writer.add_image('Image', imgs[-1], step)
        writer.add_image('Groundtruth', gts[-1], step)
        writer.add_image('Prediction', preds[-1], step)

    total_steps = len(train_loader)
    CE = torch.nn.BCEWithLogitsLoss()
    model.train()
    for step, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        imgs, gts, _, _, _ = pack
        imgs = imgs.to(device)
        gts = gts.to(device)
        if args.attention:
            atts = model(imgs)
            loss = CE(atts, gts)
            writer.add_scalar('Loss', float(loss), step)
        else:
            atts, dets = model(imgs)
            att_loss = CE(atts, gts)
            det_loss = CE(dets, gts)
            loss = att_loss + det_loss
            writer.add_scalar('Loss/Attention Loss', float(att_loss), step)
            writer.add_scalar('Loss/Detection Loss', float(det_loss), step)
            writer.add_scalar('Loss/Total Loss', float(loss), step)


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        if step % 100 == 0 or step == total_steps:
            add_image(imgs, gts, dets, step, writer)
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                  format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, args.epoch, step, total_steps, loss.data))

    save_path = 'ckpts/{}/'.format(model.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % 5 == 0:
        torch.save(model.state_dict(), '{}{}.pth.{:03d}'.format(save_path, model.name, epoch))

device = torch.device(args.device)
print('Device: {}'.format(device))

if args.attention:
    model = CPD_A().to(device)
else:
    model = CPD().to(device)
print('Model:', model.name)

optimizer = torch.optim.Adam(model.parameters(), args.lr)

transform = transforms.Compose([
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
gt_transform = transforms.Compose([
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor()])

dataset = ImageGroundTruthFolder(args.datasets_path, transform=transform, target_transform=gt_transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
writer = tensorboard.SummaryWriter(os.path.join('logs', datetime.now().strftime('%Y%m%d-%H%M%S')))
print('Dataset loaded successfully')
for epoch in range(1, args.epoch):
    print('Started epoch {:03d}/{}'.format(epoch, args.epoch))
    lr_lambda = lambda epoch: args.decay_rate ** (epoch // args.decay_epoch)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda)
    train(train_loader, model, optimizer, epoch, writer)
