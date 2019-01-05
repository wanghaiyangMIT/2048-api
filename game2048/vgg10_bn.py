import torch
import torch.nn as nn
import torchvision.models as models
import torchvision

from collections import OrderedDict

class Vgg10Conv(nn.Module):
    """
    vgg16 convolution network architecture
    """

    def __init__(self, num_cls=4, init_weights=False):
        """
        Input
            b x 1 x 4 x 4
        """
        super(Vgg10Conv, self).__init__()

        self.num_cls = num_cls
    
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(1, 64, 3, padding=1),   nn.BatchNorm2d(64),  nn.ReLU(), 
            nn.Conv2d(64, 64, 3, padding=1),  nn.BatchNorm2d(64),  nn.ReLU(), 
            #nn.MaxPool2d(2, stride=2, return_indices=True),
            # conv2
            nn.Conv2d(64, 128, 3, padding=1),  nn.BatchNorm2d(128), nn.ReLU(),   
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),  
            nn.MaxPool2d(2, stride=2, return_indices=True),
            # conv3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),  
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),  
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),  
            nn.MaxPool2d(2, stride=2, return_indices=True))

        self.classifier = nn.Sequential(
            nn.Linear(256, 512),   nn.ReLU(),  nn.Dropout(),
            nn.Linear(512, 512),          nn.ReLU(),  nn.Dropout(),
            nn.Linear(512, num_cls)
            #nn.Softmax(dim=1)
        )

        # index of conv
        self.conv_layer_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        # feature maps
        self.feature_maps = OrderedDict()
        # switch
        self.pool_locs = OrderedDict()
        # initial weight
        if init_weights:
            self.init_weights()

    def init_weights(self):
        """
        initial weights from preptrained model by vgg16
        """
        vgg16_pretrained = models.vgg16(pretrained=True)
        # fine-tune Conv2d
        for idx, layer in enumerate(vgg16_pretrained.features):
            if isinstance(layer, nn.Conv2d):
                self.features[idx].weight.data = layer.weight.data
                self.features[idx].bias.data = layer.bias.data
        # fine-tune Linear
        for idx, layer in enumerate(vgg16_pretrained.classifier):
            if isinstance(layer, nn.Linear):
                self.classifier[idx].weight.data = layer.weight.data
                self.classifier[idx].bias.data = layer.bias.data
    
    def check(self):
        model = models.vgg16(pretrained=True)
        return model

    def forward(self, x):
        """
        x.shape:        (B,C, H, W)
        return.shape:   (B , D)
        """
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
            else:
                x = layer(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
