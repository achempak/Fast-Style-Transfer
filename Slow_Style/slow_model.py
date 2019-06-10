import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as td
import torchvision as tv


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def gram_matrix(input_tensor):
    a, b, c, d = input_tensor.size()
    input_tensor = input_tensor.view(a*b, c*d)
    gm = torch.mm(input_tensor, input_tensor.t())
    return gm.div(a*b*c*d)
    


class Content(nn.Module):
    def __init__(self, target):
        super(Content, self).__init__()
        self.target = target.detach()
    
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
    
    
class Style(nn.Module):
    def __init__(self,target):
        super(Style, self).__init__()
        self.target = gram_matrix(target).detach()
        
    def forward(self, input):
        self.loss = F.mse_loss(gram_matrix(input), self.target)
        return input
    
#This layer is used to renormalize the image after every pass through the network. Necessary since
#every new image through the network is really the same image but altered.
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std
    
    
    
class Model(nn.Module):
    def __init__(self, content_target, style_target, normalization_mean, normalization_std):
        super(Model, self).__init__()

        CONTENT_LAYERS = ['conv_4']
        STYLE_LAYERS = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        DEEPEST_LAYER = 5 #deepest layer from which content or style loss is accounted for

        self.content_target = content_target
        self.style_target = style_target
        
        #Used to keep track of loss in each RUN (not whole experiment)
        self.content_loss = []
        self.style_loss = []
        
        #These two vars used for graphing purposes
        self.running_content_loss = []
        self.running_style_loss = []

        vgg = tv.models.vgg19(pretrained=True).features.to(device).eval()
        model = nn.Sequential(Normalization(normalization_mean, normalization_std)).to(device)
        
        i = 1 #this is to keep track of convolutional layers
        for layer in vgg:
            if isinstance(layer, nn.Conv2d) and i<=DEEPEST_LAYER:
                name = 'conv_{}'.format(i)
                model.add_module(name,layer)
                if(name in CONTENT_LAYERS):
                    name_content = 'content_{}'.format(i)
                    target = model(content_target).detach()
                    model.add_module(name_content, Content(target=target))
                if(name in STYLE_LAYERS):
                    name = 'style_{}'.format(i)
                    target = model(style_target).detach()
                    model.add_module(name, Style(target=target))
                i += 1
            elif isinstance(layer, nn.MaxPool2d) and i<DEEPEST_LAYER:
                name = 'avg_pool_{}'.format(i)
                model.add_module(name,nn.AvgPool2d(3, 1, 1))
            elif isinstance(layer, nn.ReLU) and i<DEEPEST_LAYER:
                name = 'relu_{}'.format(i)
                model.add_module(name,nn.ReLU(inplace=False))
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
                model.add_module(name,layer)
        print(model)
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        h = x
        self.content_loss = []
        self.style_loss = []
        for layer in self.model:
            if isinstance(layer, Content):
                h = layer.forward(h)
                c_loss = layer.loss
                self.content_loss.append(c_loss)
            elif isinstance(layer, Style):
                h = layer.forward(h)
                s_loss = layer.loss
                self.style_loss.append(s_loss)
            else:
                h = layer(h)          
                