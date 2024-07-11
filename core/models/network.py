import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class ResBackBone(nn.Module):

    def __init__(self):
        super(ResBackBone, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.in_features = resnet.fc.in_features
        self.fc = resnet.fc  # static classifier
        del resnet

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(-1, self.in_features)
        return x

    def get_output_dim(self):
        return self.in_features


class Embedding(nn.Module):

    def __init__(self, z_dim=256):
        super(Embedding, self).__init__()

        self.z_dim = z_dim
        self.bn = nn.BatchNorm1d(2048)
        self.linear = nn.Linear(2048, self.z_dim)
        # self.linear2 = nn.Linear(32 * 7, self.z_dim)

    def forward(self, hyper_net, z):
        # (N, 2048)
        z = self.bn(z)
        z = F.relu(self.linear(z.view(-1, 2048)))
        z = torch.mean(z, dim=0)  # (N, z_dim) -> (z_dim)
        w, b = hyper_net(z.view(-1, self.z_dim))
        return [w, b]


class HyperNetwork(nn.Module):

    def __init__(self, out_size, in_size, z_dim=256):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.out_size = out_size
        self.in_size = in_size

        self.w1 = Parameter(torch.Tensor(self.z_dim, self.out_size * self.in_size))
        self.b1 = Parameter(torch.Tensor(self.out_size * self.in_size))
        self.w2 = Parameter(torch.Tensor(self.z_dim, self.out_size))
        self.b2 = Parameter(torch.Tensor(self.out_size))

        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.constant_(self.b1, 0)
        nn.init.constant_(self.b2, 0)

    def forward(self, z):
        # z: (N, z_dim)
        h_final = torch.matmul(z, self.w1) + self.b1  # (1, out_size * in_size)
        w = h_final.view(self.in_size, self.out_size)
        b = torch.matmul(z, self.w2) + self.b2
        return [w, b]


class ResNetFc(nn.Module):

    def __init__(self, class_num=1000, cfg=None):
        super(ResNetFc, self).__init__()
        self.z_dim = cfg.NETWORK.Z_DIM
        self.cfg = cfg
        self.in_size = 2048
        self.out_size = class_num
        self.hyper = HyperNetwork(out_size=self.out_size, in_size=self.in_size, z_dim=self.z_dim)

        self.embed = Embedding(z_dim=self.z_dim)
        self.resnet = ResBackBone()
        self.global_avg = self.resnet.avgpool

    def forward(self, x, return_feat=False):
        x = self.resnet(x)  # (N, 2048, 7, 7)
        # x = self.global_avg(x)
        # x = x.view(-1, 2048) # (N, 2048)
        feature = x
        static_out = self.resnet.fc(x)  # static classifier
        # x = self.bn(x)
        # x = F.relu(x)
        w, b = self.embed(self.hyper, x)
        dynamic_out = torch.matmul(x, w) + b  # dynamic classifier

        x = dynamic_out
        # x = 0.5 * static_out + 0.5 * dynamic_out  # fusion
        if return_feat:
            return x, feature
        else:
            return x

    def get_param(self, initial_lr=0.1):
        params = [
            {'params': self.resnet.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.embed.parameters(), 'lr': 1.0 * initial_lr},
            {'params': self.hyper.parameters(), 'lr': 1.0 * initial_lr},
        ]
        return params
