import os
import getpass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as td
import torchvision as tv
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from collections import OrderedDict 
from collections import namedtuple
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Custom_Data_Loader(td.Dataset):
    def __init__(self, root_dir, mode = "train2014", image_size = (256, 256)):
        super(Custom_Data_Loader, self).__init__()
        self.mode = mode
        self.image_path = os.path.join(root_dir, mode)
        self.image_list = [name for name in os.listdir(self.image_path) if os.path.isfile(self.image_path + '/' + name)]
        self.image_size = image_size
        
    def __len__(self):
        return len(self.image_list)
    
    def __repr__(self):
        return "Dataset(mode = {}, image_size = {})". \
            format(self.mode, self.image_size)
    
    def __getitem__(self, idx): # Returns an image tensor corresponding to a particular file name
        image_file_name = self.image_list[idx]
        image_file_path = os.path.join(self.image_path, image_file_name)
        image = Image.open(image_file_path).convert('RGB')
        imageTransform = tv.transforms.Compose([
                     tv.transforms.Resize(self.image_size),
                     tv.transforms.CenterCrop(self.image_size),
                     tv.transforms.ToTensor(),
        ])
        x = imageTransform(image)
        return x
    

def prep_style(style_img_path, img_size=(256, 256)):
    img = Image.open(style_img_path).convert('RGB')
    transform = tv.transforms.Compose([
        tv.transforms.Resize(img_size),
        tv.transforms.CenterCrop(img_size),
        tv.transforms.ToTensor(),
    ])
    x = transform(img).unsqueeze(0) #fake batch dimension
    return x.to(device, torch.float)

def imshow(img_tensor, ax=plt):
    image = img_tensor.squeeze(0)
    image = image.to('cpu').detach().numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image)
    
def save_image(img_tensor, file_name):
    transform = tv.transforms.Compose([
        tv.transforms.ToPILImage()
    ])
    image = img_tensor.squeeze(0).to('cpu').detach()
    out = transform(image)
    out.save(file_name)
    
def save_model(model, optimizer, epoch, content_loss, style_loss,total_loss, path):
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'style_loss': style_loss,
            'content_loss': content_loss,
            'total_loss': total_loss
    }, path)
    
def load_model(model, optimizer, path, mode='eval'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    c_loss = checkpoint['content_loss']
    s_loss = checkpoint['style_loss']
    t_loss = checkpoint['total_loss']
    model.to(device)
    if mode is 'eval':
        model.eval()
    else:
        model.train()
    return epoch, c_loss, s_loss, t_loss

def plot(fig, axes, style_img, content_img, transfer_img, style_loss, content_loss, total_loss, iteration):
    axes[0][0].clear()
    axes[0][1].clear()
    axes[1][0].clear()
    axes[1][1].clear()
    axes[2][0].clear()
    axes[2][1].clear()
    imshow(content_img, ax=axes[0][0])
    axes[0][0].set_title('Content Image')

    imshow(style_img, ax=axes[0][1])
    axes[0][1].set_title('Style Image')
    
    imshow(transfer_img, ax=axes[1][0])
    axes[1][0].set_title('Transfer Image')
    
    #assume plotting begins at iteration 20 for these plots. Adjust parameters accordingly if
    #this is changed.
    axes[1][1].plot([k for k in range(20, iteration)], style_loss,
                 label="style loss")
    axes[2][0].plot([k for k in range(20, iteration)], content_loss,
                   label="content loss")
    axes[2][1].plot([k for k in range(20, iteration)], total_loss,
                   label="total loss")
    axes[1][1].set(xlabel='Iteration',ylabel='Loss')
    axes[1][1].legend()
    axes[2][0].set(xlabel='Iteration',ylabel='Loss')
    axes[2][0].legend()
    axes[2][1].set(xlabel='Iteration',ylabel='Loss')
    axes[2][1].legend()
    plt.tight_layout()
    fig.canvas.draw()
