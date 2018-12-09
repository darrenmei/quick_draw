import torch
import torch.nn as nn
import torch.nn.functional as F

# Model based on CellNuclei.ipynb reference [1]

class CNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 7, padding = 0)
        nn.init.kaiming_normal_(self.conv11.weight)
        self.pool12 = nn.MaxPool2d(kernel_size = 3, stride = 1)

        # 20 x 20 x 64 channels
        # w/out padding 20 x 20
        pad2 = samePad(5, 1)
        self.conv21 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5, padding = 0)
        nn.init.kaiming_normal_(self.conv21.weight)
        self.pool22 = nn.MaxPool2d(kernel_size = 3, stride = 1)

        # 18 x 18 x 128 channels
        # w/out padding 14 x 14
        pad3 = samePad(3, 1)
        self.conv31 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 0)
        nn.init.kaiming_normal_(self.conv31.weight)
        self.conv32 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 0)
        nn.init.kaiming_normal_(self.conv32.weight)
        self.pool33 = nn.MaxPool2d(kernel_size = 3, stride = 1)

        # 16 x 16 x 512
        # w/out padding 8 x 8
        pad4 = samePad(5, 1)
        self.conv41 = nn.Conv2d(in_channels = 512, out_channels = 4096, kernel_size = 5, padding = 0)
        nn.init.kaiming_normal_(self.conv41.weight)
        self.pool42 = nn.MaxPool2d(kernel_size = 3, stride = 1)

        # 2 x 2 x 4096 (Took out same padding)
        self.conv51 = nn.Conv2d(in_channels = 4096, out_channels = 4096, kernel_size = 2, padding = 0)
        nn.init.kaiming_normal_(self.conv51.weight)

        # # 2 x 2 x 4096
        # self.conv71 = nn.Conv2d(in_channels = 4096, out_channels = 4096, kernel_size = 2, padding = 0)
        # nn.init.kaiming_normal_(self.conv71.weight)

        # 1 x 1 x 250 (Took out same padding)
        self.conv61 = nn.Conv2d(in_channels = 4096, out_channels = 250, kernel_size = 1, padding = 0)
        nn.init.kaiming_normal_(self.conv61.weight)
        #self.output_fn = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv11(x)
        x2 = self.pool12(F.relu(x1))

        x3 = self.conv21(x2)
        x4 = self.pool22(F.relu(x3))

        x5 = self.conv31(x4)
        x6 = self.conv32(F.relu(x5))
        x7 = self.pool33(F.relu(x6))

        x8 = self.conv41(x7)
        x9 = self.pool42(F.relu(x8))

        x10 = self.conv51(x9)

        x_out = self.conv61(x10)

        # x_out = F.softmax(x11)
        # x_out = self.output_fn(x11)

        return x_out

def samePad(filterSize, stride):
    return int(float(filterSize - stride)/2)
