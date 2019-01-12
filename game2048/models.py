from torch.autograd import Variable
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from Resnet import Resnet,BasicBlock,Bottleneck
from vgg10_bn import Vgg10Conv
class MyAgent(nn.Module):
    '''
    input shape B x C x H x W
    output shape 1
    '''
    def __init__(self,args,batch_size):

        super(MyAgent,self).__init__()
        self.vgg10_bn = Vgg10Conv()
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size

    def loss(self,x,label):
        x = self.vgg10_bn(x)
        loss = self.criterion(x,label)
        return loss
    
    def forward(self,x):
        x = self.vgg10_bn(x)
        
        return x



        

