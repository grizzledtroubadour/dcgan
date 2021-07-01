from torchvision import transforms
import torch


def get_transform(type):
    if type=='simple_transform':
        transf = transforms.Compose([transforms.Resize(300)
                                     ,transforms.CenterCrop((256,256))
                                     ,transforms.ToTensor()
                                     ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    if type=='non_transform':
        transf = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop((256,256)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    return transf
