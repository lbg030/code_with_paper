# Alexnet 구현
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels = 96, kernel_size=11, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=0.5) # p의 default = 0.5다.
        self.relu = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(2,2)
        self.maxpool2 = nn.MaxPool2d(2,2)
        self.maxpool3 = nn.MaxPool2d(2,2)

    def forward(self,x):
        x1 = self.maxpool1(self.relu(self.conv1(x)))
        x2 = self.maxpool2(self.relu(self.conv2(x1)))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))

        x = self.maxpool3(x5)
        x = self.dropout(x)

        return x
    
model = AlexNet()
alexnet_dummy_input = torch.randn((1, 3, 227,227))

alexnet_dummy_output = model(alexnet_dummy_input)

print(alexnet_dummy_output.shape)

print("END")

