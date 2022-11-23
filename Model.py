from torch.nn import functional as F
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),

            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),

            nn.Linear(500, 3),
            nn.BatchNorm1d(3),
            nn.Sigmoid(),

        )

        self.decode = nn.Sequential(
            nn.Linear(3, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),

            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),

            nn.Linear(500, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Sigmoid(),

        )

    def forward(self, x):
        enOutputs = self.encode(x)
        outputs = self.decode(enOutputs)

        return enOutputs, outputs


class FCNetwork_SPAT_spec(nn.Module):
    def __init__(self, input_dim, args):
        super(FCNetwork_SPAT_spec, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),

            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),

            nn.Linear(500, 3),
            nn.BatchNorm1d(3),
            nn.Sigmoid(),

        )

        self.conv1 = nn.Conv2d(3, 100, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(100)

        self.conv2 = nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(100)

        self.conv2_1 = nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(100)

        self.conv2_2 = nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(100)

        self.conv2_3=nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1)
        self.bn2_3= nn.BatchNorm2d(100)

        self.conv3 = nn.Conv2d(100, args.nChannel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(args.nChannel)

        self.umsample = nn.UpsamplingNearest2d(scale_factor=2)


    def forward(self, x, m, n, k):
        x = x.view(k, -1)
        x = x.permute(1, 0)
        x_ = self.encode(x)
        x = x_.view(-1, m, n, 3)

        x = x.permute(0, 3, 1, 2)

        x = self.umsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x_ = F.relu(x)

        x = self.conv2(x_)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = F.relu(x)

        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = F.relu(x)

        x3 = self.conv3(x)
        x = self.bn3(x3)

        return x[0], x_, x3

class FCNetwork_spat_SPEC(nn.Module):
    def __init__(self, input_dim, args):
        super(FCNetwork_spat_SPEC, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),

            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),

            nn.Linear(500, 3),
            nn.BatchNorm1d(3),
            nn.Sigmoid(),

        )

        self.conv1 = nn.Conv2d(3, 100, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(100)

        self.conv2 = nn.Conv2d(100, 100, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(100)

        self.conv2_1 = nn.Conv2d(100, 100, kernel_size=1, stride=1, padding=0)
        self.bn2_1 = nn.BatchNorm2d(100)

        self.conv2_2 = nn.Conv2d(100, 100, kernel_size=1, stride=1, padding=0)
        self.bn2_2 = nn.BatchNorm2d(100)

        self.conv2_3=nn.Conv2d(100, 100, kernel_size=1, stride=1, padding=0)
        self.bn2_3= nn.BatchNorm2d(100)

        self.conv3 = nn.Conv2d(100, args.nChannel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(args.nChannel)

        self.umsample = nn.UpsamplingNearest2d(scale_factor=2)


    def forward(self, x, m, n, k):
        x = x.view(k, -1)
        x = x.permute(1, 0)
        x_ = self.encode(x)
        x = x_.view(-1, m, n, 3)

        x = x.permute(0, 3, 1, 2)

        x = self.umsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x_ = F.relu(x)

        x = self.conv2(x_)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = F.relu(x)

        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = F.relu(x)

        x3 = self.conv3(x)
        x = self.bn3(x3)

        return x[0], x_, x3