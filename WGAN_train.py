import argparse
import os
import numpy as np
import math
import sys
import toml

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

# self-building git
from model import Generator, Discriminator, Classifier, weights_init
from utils import compute_gradient_penalty
from tools.model import load_resnet18
from tools.utils import load_config, save_config
from tools.dataset import get_transform

# 缩写
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# 训练一轮数据
def train_model(model, dataloader, optimizer, z_dim, is_train={'G':True,'D':True}):
    
    # 过程中损失值总和
    running_g_loss = 0.0
    running_d_loss = 0.0
    
    # 设置训练或验证模式
    model['G'].train(is_train['G'])
    model['D'].train(is_train['D'])

    for i, (imgs, _) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))  # shape:(batch_size, channels, image_size, image_size)

        # Sample noise as generator input.
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, z_dim))))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer['G'].zero_grad()
        optimizer['D'].zero_grad()

        # Generate a batch of images
        fake_imgs = model['G'](z)

        # fake image discriminator out
        fake_validity = model['D'](fake_imgs)

        # Adversarial loss
        g_loss = -torch.mean(fake_validity)

        if is_train['G']:
            g_loss.backward()
            optimizer['G'].step()

        running_g_loss += g_loss.item()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer['D'].zero_grad()

        # Generate a batch of images
        fake_imgs = model['G'](z)

        # real image discriminator out
        real_validity = model['D'](real_imgs)
        # fake image discriminator out
        fake_validity = model['D'](fake_imgs)

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(model['D'], real_imgs.data, fake_imgs.data, lambda_gp=10)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty

        d_loss.backward()
        optimizer['D'].step()

        running_d_loss += d_loss.item()

        
        if i % 10 == 0:
            print('[batch:{0}/{1}] [generator loss:{2}] [discriminator loss:{3}]'\
                  .format(i, len(dataloader), g_loss.item()/batch_size, d_loss.item()/batch_size) )
        
    # --------------------------
    #  Calculate the epoch loss
    # --------------------------

    epoch_g_loss = running_g_loss / len(dataloader.dataset)
    epoch_d_loss = running_d_loss / len(dataloader.dataset)
    
    return epoch_g_loss, epoch_d_loss


if __name__ == '__main__':
    # 训练参数配置
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--dataset_name", type=str, help="The name of used dataset")
    parser.add_argument("-C", "--config_file", type=str, default='WGAN_config', help="config file name")
    parser.add_argument("-R", "--resume", action="store_true", help="Resume the experiment from latest checkpoint.")
    parser.add_argument("-E", "--n_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("-B", "--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("-N", "--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("--sample_interval", type=int, default=200, help="interval between image samples")
    opt = parser.parse_args()
    print(opt)
    
    # 图像配置
    img_shape = (opt.channels, opt.img_size, opt.img_size)
    
    # 配置与路径 
    config_file = opt.config_file
    config = load_config(config_file)

    # 数据集路径
    dataset_name = opt.dataset_name
    if not dataset_name:
        dataset_name = config['data']['default_dataset_name']
    original_image_dir = os.path.join(config['data']['original_image_root'], dataset_name)
    config['checkpoints']['dataset_name'] = dataset_name

    # Configure data loader
    data_transforms = get_transform('non_transform')
    dsets = datasets.ImageFolder(root=original_image_dir, transform=data_transforms)
    dataloader = DataLoader(dsets, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
    z_dim = config['model']['args']['latent_dim'] # 隐向量维度
    
    print('----------------------------------------------------------------------')
    print('The original images will be readed at:', original_image_dir)
    # 产出保存路径
    checkpoints_dir = os.path.join(config['checkpoints']['checkpoints_root'], 'WGAN_' + dataset_name)
    sample_image_dir = os.path.join(config['checkpoints']['sample_image_root'], 'WGAN_' + dataset_name)
    print('\nThe checkpoints will be saved at: {}.'.format(checkpoints_dir))
    print('The sample images will be saved at: {}.'.format(sample_image_dir))
    print('----------------------------------------------------------------------')
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(sample_image_dir, exist_ok=True)

    # fixed noise for sample
    fixed_noise = Variable(FloatTensor(np.random.normal(0, 1, (100, z_dim))))

    # Initialize weights
    if opt.resume:
        generator = torch.load(os.path.join(checkpoints_dir, 'NetG_last.pth'))
        discriminator = torch.load(os.path.join(checkpoints_dir, 'NetD_last.pth'))
        
        begin_epoch = config['checkpoints']['break_epoch'] + 1
        results = np.load(os.path.join(checkpoints_dir, 'results.npy')).tolist()
    else:
        # Initialize generator and discriminator
        generator = Generator(z_dim)
        discriminator = Discriminator()
        #discriminator = load_resnet18(category_num=1, pretrained=True)
        
        generator.apply(weights_init)
        discriminator.apply(weights_init)
        
        begin_epoch = 1
        results = []

    if cuda:
        generator.cuda()
        discriminator.cuda()

    # Optimizers(WGAN的优化器不能带动量：adam不可用)
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=2e-4)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=2e-4)

    # ----------
    #  Training
    # ----------

    batches_done = 0
    end_epoch = begin_epoch + opt.n_epochs

    for epoch in range(begin_epoch, end_epoch):
        print('\nEpoch {}/{}'.format(epoch, end_epoch - 1), '\n' + '-'*10)
        
        # 训练一个epoch
        epoch_g_loss, epoch_d_loss = train_model({'G':generator,'D':discriminator}, dataloader, \
                                                 {'G':optimizer_G,'D':optimizer_D}, z_dim)

        print('training end: [average generator loss: %f]/[average discriminator loss: %f]' % (epoch_g_loss, epoch_d_loss))
        results.append([epoch, epoch_g_loss, epoch_d_loss])
        
        
        # save the samples image
        if epoch % 1 == 0:
            # generate fixed image
            fake_imgs = generator(fixed_noise)

            sample_save_path = os.path.join(sample_image_dir, "fake_samples_ep{}.png".format(epoch))
            save_image(fake_imgs.data[:30], sample_save_path, nrow=6, normalize=True)

            print('samples save as:', sample_save_path)


        # save the checkpoint model
        if epoch % 5 == 0:

            G_save_path = os.path.join(checkpoints_dir, 'NetG_ep%d.pth' % epoch)
            D_save_path = os.path.join(checkpoints_dir, 'NetD_ep%d.pth' % epoch)

            torch.save(generator.state_dict(), G_save_path)
            torch.save(discriminator.state_dict(), D_save_path)

            print('generator state dict out:', G_save_path)
            print('discriminator state dict out:', D_save_path)

            # checkpoint save
            torch.save(generator, os.path.join(checkpoints_dir, 'NetG_last.pth'))
            torch.save(discriminator, os.path.join(checkpoints_dir, 'NetD_last.pth'))
            config['checkpoints']['break_epoch'] = epoch
            save_config(config, config_file) # 警告：此处进行了对配置文件的操作

            
            statistic_save_path = os.path.join(checkpoints_dir, 'results.npy')
            np.save(statistic_save_path, results)
            print('final statistics restore:', statistic_save_path)




