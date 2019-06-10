import os
import numpy as np
import torch
from torch import nn
import torchvision as tv
from PIL import Image
from matplotlib import pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def prep(img_path1, img_path2, img_size=(512, 512)):
    img1 = Image.open(img_path1).convert('RGB')
    img2 = Image.open(img_path2).convert('RGB')
    transform = tv.transforms.Compose([
        tv.transforms.Resize(img_size),
        tv.transforms.ToTensor(),
    ])
    x = transform(img1).unsqueeze(0) #fake batch dimension
    y = transform(img2).unsqueeze(0)
    return x.to(device, torch.float), y.to(device, torch.float)


def imshow(img_tensor, ax=plt):
    image = img_tensor.to('cpu')
    image = image.squeeze(0).detach().numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image)
    
    
def plot(fig, axes, style_img, content_img, transfer_image, style_loss, content_loss, iteration):
    axes[0][0].clear()
    axes[0][1].clear()
    axes[1][0].clear()
    axes[1][1].clear()
    imshow(content_img, ax=axes[0][0])
    axes[0][0].set_title('Content Image')

    imshow(style_img, ax=axes[0][1])
    axes[0][1].set_title('Style Image')
    
    imshow(transfer_image, ax=axes[1][0])
    axes[1][0].set_title('Transfer Image')
    
    #assume plotting begins at iteration 10 for these plots. Adjust parameters accordingly if
    #this is changed.
    axes[1][1].plot([k for k in range(10, iteration)], style_loss,
                 label="style loss")
    axes[1][1].plot([k for k in range(10, iteration)], content_loss,
                   label="content loss")
    axes[1][1].set(xlabel='Epoch',ylabel='Loss')
    axes[1][1].legend()
    plt.tight_layout()
    fig.canvas.draw()