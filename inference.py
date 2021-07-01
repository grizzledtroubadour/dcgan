import os
import numpy as np
import toml
import argparse
import sys

import torchvision.utils as vutils
import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import datasets,models,transforms
from torch.autograd import Variable

# self-building git
from model import Generator, Discriminator, Classifier, weights_init
from tools.utils import load_config, save_config

# 训练参数配置
parser = argparse.ArgumentParser()
parser.add_argument("-G", "--GAN_net", type=str, default='WGAN', help="name of used GAN-nets")
parser.add_argument("-C", "--img_class", type=str, default='all', help="class of the generated image")
parser.add_argument("-N", "--img_num", type=int, default=1000, help="number of the generated image")
parser.add_argument("--batch_size", type=int, default=50, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=10, help="number of cpu threads to use during batch generation")
opt = parser.parse_args()
print(opt)

# 缩写
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# 图片保存路径配置
config_file = 'WGAN_config'
config = load_config(config_file)
generated_image_dir = os.path.join(config['data']['generated_image_root'], opt.GAN_net)
os.makedirs(generated_image_dir, exist_ok=True)

# 获取类别->序号
original_image_dir = config['data']['melenoma_image_dir']
dsets = datasets.ImageFolder(root=original_image_dir)
class_idx = dsets.class_to_idx # 类别与序号映射
n_class = len(class_idx)

# 要生成图片的类别列表
if opt.img_class == 'all':
    class_list = class_idx.keys()
else:
    class_list = [opt.img_class]


z_dim = config['model']['args']['latent_dim'] #generator.latent_dim


# 使用的生成网络
#generator = Generator(z_dim, n_class)
generator = Generator(z_dim)
Gen_state_save_path = os.path.join(config['checkpoints']['checkpoints_root'], opt.GAN_net, 'NetG_ep480.pth')
generator.load_state_dict(torch.load(Gen_state_save_path))

# 使用的判别网络
discriminator = Discriminator()
Dis_state_save_path = os.path.join(config['checkpoints']['checkpoints_root'], opt.GAN_net, 'NetD_ep640.pth')
discriminator.load_state_dict(torch.load(Dis_state_save_path))

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()


# INFERENCE
generator.eval()
discriminator.eval()

l = {}
for c in class_list:
    os.makedirs(os.path.join(generated_image_dir, c), exist_ok=True)
    process_num = 0
    a = 0
    while True:

        # Sample noise as generator input with its label.
        z = Variable(FloatTensor(np.random.normal(0, 1, (opt.batch_size, z_dim))))
        #gen_labels = Variable(LongTensor(opt.batch_size).fill_(class_idx[c]))

        #fake_imgs = generator(z, gen_labels)
        fake_imgs = generator(z)
        fake_validity = discriminator(fake_imgs).data

        print(process_num)
        a += sum(fake_validity > 0.5)

        for i in range(fake_imgs.size(0)):
            if fake_validity[i] > -10000: #0.5:
                process_num += 1
                img_save_path = os.path.join(generated_image_dir, c, 'fake_{0}.jpg'.format(process_num))
                vutils.save_image(fake_imgs[i,:,:,:], img_save_path, normalize=True)
            if process_num == opt.img_num:
                break

        if process_num == opt.img_num:
            print('{:d} image of {} saved'.format(process_num, c))
            break

    l[c] = a
    
print(l)



    




