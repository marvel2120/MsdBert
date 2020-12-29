import torch.nn as nn
import torch.nn.functional as F


class myResnet(nn.Module):
    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, x, att_size=7):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # x = self.avgpool(x) [1,2048,1,1]
        # 1*2048*7*7
        fc = x.mean(3).mean(2)
        # 1*2048
        att = F.adaptive_avg_pool2d(x, [att_size, att_size])
        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        # x= Variable(x.data)
        # fc = Variable(fc.data)
        # # batchsize*2048*7*7
        # att = Variable(att.data)
        return x, fc, att
