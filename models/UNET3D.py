import torch 
import torch.nn as nn
import torch.nn.functional as F


class Down_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down_block, self).__init__()

        self.out_channels = out_channels

        self.conv = nn.Conv3d(in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, input):
        layer = F.relu(self.bn(self.conv(input)))

        conv = F.relu(self.bn2(self.conv2(layer)))

        return conv

class Up_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up_block, self).__init__()

        self.out_channels = out_channels

        self.conv = nn.Conv3d(in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, input):
        layer = F.relu(self.bn(self.conv(input)))

        conv = F.relu(self.bn2(self.conv2(layer)))

        return conv


class UNet(nn.Module):
    def __init__(self, num_channels=1, num_outs=3):
        super(UNet, self).__init__()

        self.down1 = Down_block(in_channels=num_channels, out_channels=num_channels*16)
        self.down2 = Down_block(in_channels=num_channels*16, out_channels= num_channels*32)
        self.down3 = Down_block(in_channels=num_channels* 32, out_channels= num_channels *64)
        self.down4 = Down_block(in_channels=num_channels* 64, out_channels=num_channels * 128)

        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.trans4 = nn.ConvTranspose3d(in_channels= num_channels *128, out_channels=num_channels*64, kernel_size=2, stride=2)
        self.trans3 = nn.ConvTranspose3d(in_channels= num_channels *64, out_channels=num_channels*32, kernel_size=2, stride=2)
        self.trans2 = nn.ConvTranspose3d(in_channels= num_channels *32, out_channels=num_channels*16, kernel_size=2, stride=2)
        
        

        self.up3 = Up_block(in_channels=num_channels*128, out_channels=num_channels*64)
        self.up2 = Up_block(in_channels=num_channels*64, out_channels= num_channels*32)
        self.up1 = Up_block(in_channels=num_channels* 32, out_channels= num_channels *16)

        self.final_conv = nn.Conv3d(in_channels=num_channels*16, out_channels=num_outs, kernel_size=1)

    def forward(self, input):

        layer1 = self.down1(input)
        max_pool1 = self.max_pool(layer1)

        layer2 = self.down2(max_pool1)
        max_pool2 = self.max_pool(layer2)

        layer3 = self.down3(max_pool2)
        max_pool3 = self.max_pool(layer3)
   
        layer4 = self.down4(max_pool3)
        
      

        trans4 = self.trans4(layer4)
        cat4 = torch.cat([trans4, layer3], dim=1)
        up3 = self.up3(cat4)
       

        trans3 = self.trans3(up3)
        cat3 = torch.cat([trans3, layer2], dim=1)
        up2 = self.up2(cat3)
      

        trans2 = self.trans2(up2)
        cat2 = torch.cat([trans2, layer1], dim=1)
        up1 = self.up1(cat2)
      
        final_layer = self.final_conv(up1)

        return final_layer


