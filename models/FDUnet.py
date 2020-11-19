import torch
import torch.nn as nn
import torch.nn.functional as F


class block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(block, self).__init__()
        
        self.conv11 = nn.Conv3d(in_channels,out_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm3d(out_channels)
        out_channels2 = int(out_channels/2)
        self.conv33 = nn.Conv3d(out_channels, out_channels2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels2)

    def forward(self, input):

        conv1 = F.relu(self.bn(self.conv11(input)))

        conv2 = F.relu(self.bn1(self.conv33(conv1)))


        return  conv2


class Dense_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dense_block, self).__init__()
        
        incr = int(in_channels/2)
        self.dense_block1 = block(in_channels, in_channels)
        self.dense_block2 = block(in_channels + incr, in_channels)
       # self.dense_block3 = block(in_channels + 2* incr, in_channels)
        #self.dense_block4 = block(in_channels + 3* incr, in_channels)


    def forward(self, input):


        conv1 = self.dense_block1(input) #16
        cat1 = torch.cat([input, conv1], dim=1) #48
        conv2 = self.dense_block2(cat1) #16
        cat2 = torch.cat([cat1, conv2], dim=1) #64 #

        # if you want to add more layer at dense block, you can uncomment the following lines 
        # and change incr in incr = int(in_channels/4), and uncomment the other dense_block in init
        # finally, modify out_channels2/2 in out_channels/4 in class block

        #conv3 = self.dense_block3(cat2) #8
        #cat3 = torch.cat([cat2, conv3], dim=1) #56
        #conv4 = self.dense_block4(cat3) # 8
        #cat4 = torch.cat([cat3,conv4], dim=1) #64

        return cat2




class Down_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down_block, self).__init__()

        self.out_channels = out_channels

        self.conv = nn.Conv3d(in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)


    def forward(self, input):
        
        layer = F.relu(self.bn(self.conv(input)))

        return layer


class Up_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up_block, self).__init__()

        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

        self.bn = nn.BatchNorm3d(out_channels)



    def forward(self, input):

        layer = self.deconv(input)
        layer = self.bn(layer)
        layer = F.relu(layer)


        return layer
        


class conv11(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv11, self).__init__()


        self.conv11 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, input):

        layer = F.relu(self.bn(self.conv11(input)))


        return layer






class FDUnet(nn.Module):
    def __init__(self, num_channels=1, num_outs=3):
        super(FDUnet, self).__init__()


        self.down_conv1 = Down_block(in_channels = num_channels, out_channels=32)
        self.layer1 = Dense_block(in_channels=32, out_channels=64)
        self.layer2 = Dense_block(in_channels=64, out_channels=128)
        self.layer3 = Dense_block(in_channels=128, out_channels=256)
        self.layer4 = Dense_block(in_channels=256, out_channels=512)

        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)


        self.up4 = Up_block(in_channels=512, out_channels=256)
        self.up3 = Up_block(in_channels=256, out_channels=128)
        self.up2 = Up_block(in_channels=128, out_channels=64)

        self.conv4 = conv11(in_channels=512, out_channels=128)
        self.conv3 = conv11(in_channels=256, out_channels=64)
        self.conv2 = conv11(in_channels=128, out_channels=32)

        self.final_conv = nn.Conv3d(in_channels=64, out_channels=num_outs, kernel_size=1)



    def forward(self, input):



        down_conv1 = self.down_conv1(input) #32

        layer1 = self.layer1(down_conv1)
        pool1 = self.max_pool(layer1)
        layer2 = self.layer2(pool1)
        pool2 = self.max_pool(layer2)
        layer3 = self.layer3(pool2)
        pool3 = self.max_pool(layer3)
        layer4 = self.layer4(pool3)





        up4 = self.up4(layer4)
        cat4 = torch.cat([up4, layer3], dim=1)
        conv4 = self.conv4(cat4)
        dense4 = self.layer3(conv4)

        up3 = self.up3(dense4)
        cat3 = torch.cat([up3, layer2], dim=1)
        conv3 = self.conv3(cat3)
        dense3 = self.layer2(conv3)

        up2 = self.up2(dense3)
        cat2 = torch.cat([up2, layer1], dim=1)
        conv2 = self.conv2(cat2)
        dense2 = self.layer1(conv2)

        final_conv = self.final_conv(dense2)


        return final_conv




        







        



        