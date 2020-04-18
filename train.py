"""
-------------------------------------
denoising
-------------------------------------
python  train.py \
        --mode train \
        --network unet \
        --learning_type residual \
        --task denoising \
        --opts random 30.0

python  train.py \
        --mode train \
        --network resnet \
        --learning_type residual \
        --task denoising \
        --opts random 30.0

-------------------------------------
inpainting
-------------------------------------
python  train.py \
        --mode train \
        --network unet \
        --learning_type residual \
        --task inpainting \
        --opts random 0.5

python  train.py \
        --mode train \
        --network resnet \
        --learning_type residual \
        --task inpainting \
        --opts random 0.5

-------------------------------------
super_resolution
-------------------------------------
python  train.py \
        --mode train \
        --network unet \
        --learning_type residual \
        --task super_resolution \
        --opts bilinear 4.0

python  train.py \
        --mode train \
        --network resnet \
        --learning_type residual \
        --task super_resolution \
        --opts bilinear 4.0

python  train.py \
        --mode train \
        --network srresnet \
        --learning_type residual \
        --task super_resolution \
        --opts bilinear 4.0 0.0
"""

## 라이브러리 추가하기
import argparse

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from dataset import *
from util import *

import matplotlib.pyplot as plt

from torchvision import transforms

## Parser 생성하기
parser = argparse.ArgumentParser(description="Regression Tasks such as inpainting, denoising, and super_resolution",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--mode", default="train", choices=["train", "test"], type=str, dest="mode")
parser.add_argument("--train_continue", default="on", choices=["on", "off"], type=str, dest="train_continue")

parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=128, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=300, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="../../datasets/img_align_celeba", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint/DCGAN", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log/DCGAN", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result/DCGAN", type=str, dest="result_dir")

parser.add_argument("--task", default="GAN", choices=["inpainting", "denoising", "super_resolution"], type=str, dest="task")
parser.add_argument('--opts', nargs='+', default=['bilinear', 4.0, 0], dest='opts')

parser.add_argument("--ny", default=64, type=int, dest="ny")
parser.add_argument("--nx", default=64, type=int, dest="nx")
parser.add_argument("--nch", default=3, type=int, dest="nch")
parser.add_argument("--nker", default=128, type=int, dest="nker")

parser.add_argument("--network", default="DCGAN", choices=["unet", "hourglass", "resnet", "srresnet", "DCGAN"], type=str, dest="network")
parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")

args = parser.parse_args()

## 트레이닝 파라메터 설정하기
mode = args.mode
train_continue = args.train_continue

lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

task = args.task
opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

ny = args.ny
nx = args.nx
nch = args.nch
nker = args.nker

network = args.network
learning_type = args.learning_type

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("mode: %s" % mode)

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)

print("task: %s" % task)
print("opts: %s" % opts)

print("network: %s" % network)
print("learning type: %s" % learning_type)

print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)

print("device: %s" % device)

## 디렉토리 생성하기
result_dir_train = os.path.join(result_dir, 'train')
result_dir_test = os.path.join(result_dir, 'test')

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir_train, 'png'))
    # os.makedirs(os.path.join(result_dir_train, 'numpy'))

    os.makedirs(os.path.join(result_dir_test, 'png'))
    os.makedirs(os.path.join(result_dir_test, 'numpy'))

## 네트워크 학습하기
if mode == 'train':
    transform_train = transforms.Compose([Resize(shape=(ny, nx)), Normalization(mean=0.5, std=0.5)])
    dataset_train = Dataset(data_dir=data_dir, transform=transform_train, task=task, opts=opts)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

    # 그밖에 부수적인 variables 설정하기
    num_data_train = len(dataset_train)
    num_batch_train = np.ceil(num_data_train / batch_size)

## 네트워크 생성하기
if network == "unet":
    net = UNet(in_channels=nch, out_channels=nch, nker=nker, learning_type=learning_type).to(device)
elif network == "hourglass":
    net = Hourglass(in_channels=nch, out_channels=nch, nker=nker, learning_type=learning_type).to(device)
elif network == "resnet":
    net = ResNet(in_channels=nch, out_channels=nch, nker=nker, learning_type=learning_type).to(device)
elif network == "srresnet":
    net = SRResNet(in_channels=nch, out_channels=nch, nker=nker, learning_type=learning_type).to(device)
elif network == "DCGAN":
    netG = DCGAN(in_channels=100, out_channels=nch, nker=nker).to(device)
    netD = Discriminator(in_channels=nch, out_channels=1, nker=nker).to(device)

    init_weights(netG, init_type='normal', init_gain=0.02)
    init_weights(netD, init_type='normal', init_gain=0.02)

## 손실함수 정의하기
# fn_loss = nn.BCEWithLogitsLoss().to(device)
# fn_loss = nn.MSELoss().to(device)
fn_loss = nn.BCELoss().to(device)

## Optimizer 설정하기
optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

## 그밖에 부수적인 functions 설정하기
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

cmap = None if nch == 3 else "gray"

## Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))

## 네트워크 학습시키기
st_epoch = 0

# TRAIN MODE
if mode == 'train':
    if train_continue == "on":
        netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        netG.train()
        netD.train()

        loss_G_train = []
        loss_D_real_train = []
        loss_D_fake_train = []

        for batch, data in enumerate(loader_train, 1):
            # forward pass
            label = data['label'].to(device)
            input = torch.randn(label.size(0), 100, 1, 1).to(device)

            output = netG(input)

            # backward netD
            set_requires_grad(netD, True)
            optimD.zero_grad()

            pred_real = netD(label)
            pred_fake = netD(output.detach())

            loss_D_real = fn_loss(pred_real, torch.ones_like(pred_real))
            loss_D_fake = fn_loss(pred_fake, torch.zeros_like(pred_fake))
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

            loss_D.backward()
            optimD.step()

            # backward netD
            set_requires_grad(netG, False)
            optimG.zero_grad()

            pred_fake = netD(output)

            loss_G = fn_loss(pred_fake, torch.ones_like(pred_fake))

            loss_G.backward()
            optimG.step()

            # 손실함수 계산
            loss_G_train += [loss_G.item()]
            loss_D_real_train += [loss_D_real.item()]
            loss_D_fake_train += [loss_D_fake.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | "
                  "GEN: %.4f | DISC REAL: %.4f | DISC FAKE: %.4f" %
                  (epoch, num_epoch, batch, num_batch_train,
                   np.mean(loss_G_train), np.mean(loss_D_real_train), np.mean(loss_D_fake_train)))

            if batch % 20 == 0:
              # Tensorboard 저장하기
              id = num_batch_train * (epoch - 1) + batch

              output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()
              output = np.clip(output, a_min=0, a_max=1)

              plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output.png' % id), output[0], cmap=cmap)

              # writer_train.add_image('output', output, id, dataformats='NHWC')

        writer_train.add_scalar('loss_G', np.mean(loss_G_train), epoch)
        writer_train.add_scalar('loss_D_real', np.mean(loss_D_real_train), epoch)
        writer_train.add_scalar('loss_D_fake', np.mean(loss_D_fake_train), epoch)

        if epoch % 50 == 0:
            save(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD, epoch=epoch)

    writer_train.close()

# TEST MODE
else:
    netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD)

    with torch.no_grad():
        netG.eval()

        # forward pass
        input = torch.randn(batch_size, 100, 1, 1).to(device)

        output = netG(input)

        print("TEST COMPLETE")

        # Tensorboard 저장하기
        output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()

        for j in range(output.shape[0]):
            id = j
            output_ = output[j]
            output_ = np.clip(output_, a_min=0, a_max=1)

            plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output.png' % id), output_, cmap=cmap)


