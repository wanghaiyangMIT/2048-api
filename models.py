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
        #self.Resnet34 = Resnet(BasicBlock, [3, 4, 6, 3], num_classes = 4)
        #self.Resnet50 = Resnet(Bottleneck, [3, 4, 6, 3], num_classes = 4)
        #self.Resnet101 = Resnet(Bottleneck, [3, 4, 23, 3], num_classes = 4)
        self.vgg10_bn = Vgg10Conv()
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size

    def loss(self,x,label):
        #x = self.Resnet34(x)
        x = self.vgg10_bn(x)
        #x = torch.max(x,1)[1].cpu().view(-1,1)
        #x = torch.zeros(self.batch_size, 4).scatter_(1, x , 1)
        #x = x.cuda()
        #out = Variable(x,requires_grad=True)
        #label = Variable(label,requires_grad=True)
        #print(x.size(),x.dtype,x)
        #print(label.size(),label.dtype,label)
        #x = torch.max(x,1)[1]
        #print(x)
        #print(label)
        
        loss = self.criterion(x,label)
        return loss
    
    def forward(self,x):
        x = self.vgg10_bn(x)
        #x = self.Resnet34(x)
        
        return x



        

