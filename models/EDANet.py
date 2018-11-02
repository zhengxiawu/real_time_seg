#######################
# name: EDANet full model definition reproduced by Pytorch(v0.4.1)
# data: Sept 2018
# author:PengfeiWang(pfw813@gmail.com)
# paper: Efficient Dense Modules of Asymmetric Convolution for Real-Time Semantic Segmentation
#######################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super(DownsamplerBlock,self).__init__()

        self.ninput = ninput
        self.noutput = noutput

        if self.ninput < self.noutput:
            # Wout > Win
            self.conv = nn.Conv2d(ninput, noutput-ninput, kernel_size=3, stride=2, padding=1)
            self.pool = nn.MaxPool2d(2, stride=2)
        else:
            # Wout < Win
            self.conv = nn.Conv2d(ninput, noutput, kernel_size=3, stride=2, padding=1)

        self.bn = nn.BatchNorm2d(noutput)

    def forward(self, x):
        if self.ninput < self.noutput:
            output = torch.cat([self.conv(x), self.pool(x)], 1)
        else:
            output = self.conv(x)

        output = self.bn(output)
        return F.relu(output)
    

class EDABlock(nn.Module):
    def __init__(self,ninput, dilated, k = 40,dropprob = 0.02):
        super(EDABlock,self).__init__()

        #k: growthrate
        #dropprob:a dropout layer between the last ReLU and the concatenation of each module

        self.conv1x1 = nn.Conv2d(ninput, k, kernel_size=1)
        self.bn0 = nn.BatchNorm2d(k)

        self.conv3x1_1 = nn.Conv2d(k, k, kernel_size=(3, 1),padding=(1,0))
        self.conv1x3_1 = nn.Conv2d(k, k, kernel_size=(1, 3),padding=(0,1))
        self.bn1 = nn.BatchNorm2d(k)

        self.conv3x1_2 = nn.Conv2d(k, k, (3, 1), stride=1, padding=(dilated,0), dilation = dilated)
        self.conv1x3_2 = nn.Conv2d(k, k, (1,3), stride=1, padding=(0,dilated), dilation =  dilated)
        self.bn2 = nn.BatchNorm2d(k)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, x):
        input = x

        output = self.conv1x1(x)
        output = self.bn0(output)
        output = F.relu(output)

        output = self.conv3x1_1(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        output = F.relu(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        output = torch.cat([output,input],1)
        #print output.size() #check the output
        return output


class EDANet(nn.Module):
    def __init__(self, num_classes=20):
        super(EDANet,self).__init__()

        self.layers = nn.ModuleList()
        self.dilation1 = [1,1,1,2,2]
        self.dilation2 = [2,2,4,4,8,8,16,16]

        # DownsamplerBlock1
        self.layers.append(DownsamplerBlock(3, 15))

        # DownsamplerBlock2
        self.layers.append(DownsamplerBlock(15, 60))

        # EDA module 1-1 ~ 1-5
        for i in range(5):
            self.layers.append(EDABlock(60 + 40 * i, self.dilation1[i]))

        # DownsamplerBlock3
        self.layers.append(DownsamplerBlock(260, 130))

        # EDA module 2-1 ~ 2-8
        for j in range(8):
            self.layers.append(EDABlock(130 + 40 * j, self.dilation2[j]))

        # Projection layer
        self.project_layer = nn.Conv2d(450,num_classes,kernel_size = 1)

        self.weights_init()

    def weights_init(self):
        for idx, m in enumerate(self.modules()):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


    def forward(self, x):

        output = x

        for layer in self.layers:
            output = layer(output)

        output = self.project_layer(output)

        # Bilinear interpolation x8
        output = F.upsample(output,scale_factor = 8,mode = 'bilinear',align_corners=True)

        # Bilinear interpolation x2 (inference only)
        # if not self.training:
        #     output = F.interpolate(output, scale_factor=2, mode='bilinear',align_corners=True)

        return output
class InputProjectionA(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''
    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super(InputProjectionA,self).__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            #pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input

class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''
    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super(BR,self).__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output
class EDA_ESP_decoder_Net(nn.Module):
    def __init__(self, num_classes=20):
        super(EDA_ESP_decoder_Net,self).__init__()
        self.num_classes = num_classes
        self.layers = nn.ModuleList()
        self.dilation1 = [1,1,1,2,2]
        self.dilation2 = [2,2,4,4,8,8,16,16]
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)

        # DownsamplerBlock1
        self.DownsamplerBlock1 = DownsamplerBlock(3, 15)

        # DownsamplerBlock2
        self.DownsamplerBlock2 = DownsamplerBlock(15+3, 60)
        self.BR1 = BR(15+3)

        # EDA module 1-1 ~ 1-5
        self.EDABlock1_1 = EDABlock(60 + 40 * 0, self.dilation1[0])
        self.EDABlock1_2 = EDABlock(60 + 40 * 1, self.dilation1[1])
        self.EDABlock1_3 = EDABlock(60 + 40 * 2, self.dilation1[2])
        self.EDABlock1_4 = EDABlock(60 + 40 * 3, self.dilation1[3])
        self.EDABlock1_5 = EDABlock(60 + 40 * 4, self.dilation1[4])


        # DownsamplerBlock3
        self.DownsamplerBlock3 = DownsamplerBlock(260+60+3, 130)
        self.BR2 = BR(260+60+3)

        # EDA module 2-1 ~ 2-8
        self.EDABlock2_1 = EDABlock(60 + 40 * 0, self.dilation2[0])
        self.EDABlock2_2 = EDABlock(60 + 40 * 1, self.dilation2[1])
        self.EDABlock2_3 = EDABlock(60 + 40 * 2, self.dilation2[2])
        self.EDABlock2_4 = EDABlock(60 + 40 * 3, self.dilation2[3])
        self.EDABlock2_5 = EDABlock(60 + 40 * 4, self.dilation2[4])
        self.EDABlock2_6 = EDABlock(60 + 40 * 5, self.dilation2[5])
        self.EDABlock2_7 = EDABlock(60 + 40 * 6, self.dilation2[6])
        self.EDABlock2_8 = EDABlock(60 + 40 * 7, self.dilation2[7])
        self.BR3 = BR(450)


        # Projection layer
        self.project_layer = nn.Conv2d(450,num_classes,kernel_size = 1)

        self.weights_init()

    def weights_init(self):
        for idx, m in enumerate(self.modules()):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


    def forward(self, x):
        inp1 = self.sample1(x)
        inp2 = self.sample2(x)
        level_1 = self.DownsamplerBlock1(x)
        level_1_concat = self.BR1(torch.cat([level_1, inp1], 1))
        level_2 = self.DownsamplerBlock2(level_1_concat)
        x = self.EDABlock1_1(level_2)
        x = self.EDABlock1_2(x)
        x = self.EDABlock1_3(x)
        x = self.EDABlock1_4(x)
        x = self.EDABlock1_5(x)

        level_2_concat = self.DownsamplerBlock3(torch.cat([x,level_2, inp2], 1))
        x = self.DownsamplerBlock3(level_2_concat)
        x = self.EDABlock2_1(x)
        x = self.EDABlock2_2(x)
        x = self.EDABlock2_3(x)
        x = self.EDABlock2_4(x)
        x = self.EDABlock2_5(x)
        x = self.EDABlock2_6(x)
        x = self.EDABlock2_7(x)
        x = self.EDABlock2_8(x)

        x = self.project_layer(x)


        # Bilinear interpolation x8
        output = F.upsample(output,scale_factor = 8,mode = 'bilinear',align_corners=True)

        # Bilinear interpolation x2 (inference only)
        # if not self.training:
        #     output = F.interpolate(output, scale_factor=2, mode='bilinear',align_corners=True)

        return output
if __name__ == '__main__':

    input = Variable(torch.randn(1,3,512,1024))
    # for the inference only mode
    net = EDANet().eval()
    # for the training mode
    #net = EDANet().train()
    output = net(input)
    print output.size()
