import torch
import torch.nn as nn
from math import pow

class QuickDraw(nn.Module):
    def __init__(self, input_size = 28, num_classes = 15):
        super(QuickDraw, self).__init__()
        self.num_classes = num_classes
        #  pic: 3 dimension 
        # [pic pic ... pic] X Y Z 
        
        # https://phamdinhkhanh.github.io/2019/08/22/convolutional-neural-network.html
        
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 5, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(2,2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        dimension = int(64 * pow(input_size/4 - 3, 2))
        self.fc1 = nn.Sequential(nn.Linear(dimension, 512), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(512, 128), nn.Dropout(0.5))
        self.fc3 = nn.Sequential(nn.Linear(128, num_classes))

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = output.view(output.size(0), -1)  # (batch_size , dimension ) * (dimension, 512) = (batch size , 512)
        output = self.fc1(output)
        output = self.fc2(output) # batch_size * 128
        output = self.fc3(output) # batch_size * num_classes (1 pic ~ [x1, x2, ..., xn] len(num_classes) )
        return output
    
    # Apple pic -> apple ->  [1, 0, 0, 0, 0]. Vd [0.2, 0.2, 0.2, 0.2, 0.2] - optimize loss-> [0.9, 0.05 0.03 0.1 0]
    # loss([1, 0, 0, 0, 0].  [0.2, 0.2, 0.2, 0.2, 0.2] ) = 1000
    # loss([1, 0, 0, 0, 0].   [0.9, 0.05 0.03 0.1 0] = 1
    # optimizer make loss from 1000 -> 1
    
    # Ball pic -> ball -> [0, 1, 0, 0, 0]
    
    
    # new =>  logits [[0.9, 0.05 0.03 0.1 0] ]