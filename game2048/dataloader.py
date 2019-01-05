import os
import json
import time
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[.5], std=[.5])  
transform = transforms.Compose([transforms.ToTensor(), normalize])

class mydataset(data.Dataset):
    def __init__(self,data_dir):
        self.datalist = os.listdir(data_dir)
        self.data_dir = data_dir
    
    def __getitem__(self,index):
        dataname  = self.datalist[index]
        datapath  = os.path.join(self.data_dir,dataname)
        with open(datapath,'r') as f:
            data  = json.load(f)
            tabel = np.expand_dims(data['tabel'],axis = 2)
            tabel = transform(tabel)
            direction = np.zeros((1,1))
            direction.fill(data['direction'])
            #direction[int(data['direction'])] = 1
            direction = torch.tensor(direction).float()
        
        return tabel, direction
    
    def __len__(self):
        return len(self.datalist)
        
    


