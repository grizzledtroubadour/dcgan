import os
import torch.nn as nn
import torch


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(1.0, 0.02)



class Generator(nn.Module):
    def __init__(self, inplanes=10, n_class=0):
        super(Generator, self).__init__()

        self.latent_dim = inplanes
        self.n_class = n_class
        if n_class > 0:
            self.label_emb = nn.Embedding(n_class, n_class)
        
        def block(in_feat, out_feat, kernel_size, stride, padding, bias, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 
                                         kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.ReLU(True))
            return layers

        self.model = nn.Sequential(
            # input = batch_sizex10x1x1
            *block(inplanes+n_class, 256, kernel_size=4, stride=1, padding=0, bias=False),
            # Output = batch_sizex256x4x4
            *block(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            # Output = batch_sizex128x8x8
            *block(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            # Output = batch_sizex64x16x16
            *block(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            # Output = batch_sizex32x32x32
            *block(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            #output = batch_sizex16x64x64
            *block(16, 8, kernel_size=4, stride=2, padding=1, bias=False),
            #output = batch_sizex8x128x128
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
            # Output = batch_sizex3x256x256
        )
        
    def forward(self, z, labels=False):
        if self.n_class > 0:
            z = torch.cat((self.label_emb(labels), z), -1)
        inp = z.view(z.shape[0],-1,1,1)
        img = self.model(inp)
        return img

'''class Generator(nn.Module):
    def __init__(self, inplanes=100, n_class=0):
        super(Generator, self).__init__()
        
        self.n_class = n_class
        self.latent_dim = inplanes + n_class
        if n_class > 0:
            self.label_emb = nn.Embedding(n_class, 10)
        
        self.init_size = 256 // (2**6) # 初始特征图尺寸
        self.init_dim = 512 # 初始特征图维度
        self.L1 = nn.Sequential(nn.Linear(self.latent_dim, self.init_dim * (self.init_size ** 2)))
        
        def upsample_layer(in_feat, out_feat):
            layers = [nn.Upsample(scale_factor=2),
                     nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1),
                     nn.BatchNorm2d(out_feat),
                     nn.LeakyReLU(0.2, inplace=True)
                     ]
            return layers

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(self.init_dim),
            *upsample_layer(self.init_dim, 512),
            *upsample_layer(512, 256),
            *upsample_layer(256, 128),
            *upsample_layer(128, 64),
            *upsample_layer(64, 32),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels=False):
        if self.n_class > 0:
            z = torch.cat((self.label_emb(labels), z), -1)
        out = self.L1(z)
        out = out.view(out.shape[0], self.init_dim, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img'''
        

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1, bias=False),
                     nn.LeakyReLU(0.2, inplace=True)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            return layers
        
        self.model = nn.Sequential(
            # Input = batch_sizex3x256x256
            *block(3, 16, normalize=False),
            nn.LayerNorm([16, 128, 128]),
            # Output = batch_sizex16x128x128
            *block(16, 32, normalize=False),
            nn.LayerNorm([32, 64, 64]),
            # output = batch_sizex32x64x64
            *block(32, 64, normalize=False),
            nn.LayerNorm([64, 32, 32]),
            # output = batch_sizex64x32x32
            *block(64, 128, normalize=False),
            nn.LayerNorm([128, 16, 16]),
            # Output = batch_sizex128x16x16
            *block(128, 256, normalize=False),
            nn.LayerNorm([256, 8, 8]),
            # Output = batch_sizex256x8x8
            *block(256, 512, normalize=False),
            nn.LayerNorm([512, 4, 4]),
            # Output = batch_sizex512x4x4
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            #nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        validity = out.view(out.shape[0], -1)
        return validity


class Classifier(nn.Module):
    def __init__(self, n_class=2):
        super(Classifier, self).__init__()
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1, bias=False),
                     nn.LeakyReLU(0.2, inplace=True)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            return layers
        
        self.model = nn.Sequential(
            # Input = batch_sizex3x256x256
            *block(3, 16, normalize=False),
            # Output = batch_sizex16x128x128
            *block(16, 32),
            # output = batch_sizex32x64x64
            *block(32, 64),
            # output = batch_sizex64x32x32
            *block(64, 128),
            # Output = batch_sizex128x16x16
            *block(128, 256),
            # Output = batch_sizex256x8x8
            *block(256, 512),
            # Output = batch_sizex512x4x4
            nn.Conv2d(512, n_class, kernel_size=4, stride=1, padding=0),
            #nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        validity = out.view(out.shape[0], -1)
        return validity
    








    