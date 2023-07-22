import torch
import torch.nn as nn
import torch.nn.functional as F

"""
# ! Important
# * Important information is highlighted 


수식1 : Wout =  (Win(input size) - F(Filter size) + 2P(padding) / S(stride) + 1
"""


class practice_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, 7, 5
        )  # 3채널 input channel / 7채널 output channel / 5*5 filter size / 1 stride / no padding
        # * (32 - 5 + (2 * 0)) / 1 + 1 = 28
        # * --> 그렇기 때문에 (1,7,28,28)이 됨.
        self.pool = nn.MaxPool2d(2, 2)  # * (1,7, 14,14)

        self.conv2 = nn.Conv2d(7, 20, 5)
        # * (14 - 5 + (2 * 0) / 1 + 1= 10
        # * --> 그렇기 때문에 (1,20, 10, 10)이 됨.

        self.fc1 = nn.Linear(20 * 5 * 5, 200)
        # * pooling을 하고나면 (1,20,5,5)가 되기 때문에 20*5*5
        # * 200은 random으로 설정하는 것

        self.fc2 = nn.Linear(200, 120)
        self.fc3 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


my_practice_model = practice_model()

dummy_input = torch.rand((1, 3, 32, 32))

dummy_output = my_practice_model(dummy_input)

print("END")
