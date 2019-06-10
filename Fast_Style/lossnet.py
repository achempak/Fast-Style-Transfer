import numpy as np
import torch
from torch import nn
import torchvision as tv

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def gram_matrix(input_tensor):
    a, b, c, d = input_tensor.size()
    input_tensor = input_tensor.view(a, b, c*d)
    input_tensor_t = input_tensor.transpose(1, 2)
    gm = torch.bmm(input_tensor, input_tensor_t)
    return gm/(b*c*d)

class LossNet(nn.Module):
    def __init__(self):
        super(LossNet, self).__init__()
        self.model = tv.models.vgg16(pretrained=True).features.eval()
        for param in self.model.parameters():
            param.requires_grad = False
                
    def forward(self, x):
        CONTENT_LAYERS = ['conv_4']
        STYLE_LAYERS = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        style_features = []
        content_features = []
        DEEPEST_LAYER = 5
        y = x
        
        i = 1
        for layer in self.model:
            if i > DEEPEST_LAYER:
                break
            if isinstance(layer, nn.Conv2d) and i<= DEEPEST_LAYER:
                y = layer(y)
                name = 'conv_{}'.format(i)
                if(name in CONTENT_LAYERS):
                    content_features.append(y)
                if(name in STYLE_LAYERS):
                    style_features.append(y)
                i += 1
            else:
                y = layer(y)
        
        return content_features, style_features