import torch
import torch.nn.functional as F
import torchvision.utils as utils
import torchvision.transforms as transforms

import time
import numpy as np
import pdb, os, argparse
from scipy import misc

from gts_folder import ImageGroundTruthFolder
from model.CPD_models import CPD_VGG, CPD_VGG_attention
from model.CPD_ResNet_models import CPD_ResNet
from data import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--datasets_path', type=str, default='./datasets/test', help='path to datasets')
parser.add_argument('--save_path', type=str, default='./results', help='path to svae results')
parser.add_argument('--pth', type=str, default='CPD.pth', help='model filename')
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
parser.add_argument('--imgres', type=int, default=352, help='dataset image training resolution')
parser.add_argument('--attention', type=bool, default=False)
opt = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

if opt.is_ResNet:
    model = CPD_ResNet()
elif opt.attention:
    model = CPD_VGG_attention()
else:
    model = CPD_VGG()

model.load_state_dict(torch.load(opt.pth, map_location=torch.device(device)))
print('Loaded:', model.name)

if torch.cuda.is_available():
    model.cuda()
model.eval()

# test_datasets = ['PASCAL-S', 'ECSSD', 'DUT-OMRON', 'DUTS-TEST', 'HKUIS']
test_datasets = ['PASCAL-S']

for dataset in test_datasets:
    save_path = './results/{}/{}/'.format(model.name, dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = '{}/{}/imgs/'.format(opt.datasets_path, dataset)
    gt_root = '{}/{}/gts/'.format(opt.datasets_path, dataset)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    n_datapoints = test_loader.size
    n_datapoints = 10
    times = np.zeros(n_datapoints)
    for i in range(n_datapoints):
        image, gt, name = test_loader.load_data()
        print('{} - [{}/{}] {}'.format(dataset, i, test_loader.size, name))
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        if torch.cuda.is_available():
            image = image.cuda()
        t0 = time.time()
        if opt.attention:
            res = model(image)
        else:
            _, res = model(image)
        times[i] = time.time() - t0
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu()
        utils.save_image(res, save_path+name)

    print('{}'.format(1/times.mean()))
