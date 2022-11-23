import torch.nn as nn
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
import torchvision
from torch.nn import functional as F
import torch
import torch.nn as nn
import cv2
import umap
import torch
from sklearn.cluster import KMeans
import os
import glob
import scipy.io as scio
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import numpy as np
import torch.nn.init
from skimage import segmentation

import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import argparse
import glob
import os
from sklearn.metrics import adjusted_rand_score
import torch.nn as nn
from torchvision import models


class TripletLoss(nn.Module):

    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputss, targetss,uu):

        batch_size = 1000
        mm = len(inputss[:,0])
        iter = mm//batch_size
        losss = 0

        torch.manual_seed(uu)
        shuf = torch.randperm(mm)
        inputss = inputss[shuf]
        targetss = targetss[shuf]


        for i in range(5):

            if batch_size*(i+1) < mm:
                inputs = inputss[batch_size * i: batch_size*(i + 1)]
                targets = targetss[batch_size * i: batch_size * (i + 1)]
                n = inputs.size(0)

                dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
                dist = dist + dist.t()
                dist.addmm_(1, -2, inputs, inputs.t())
                dist = dist.clamp(min=1e-12).sqrt()

                mask = targets.expand(n, n).eq(targets.expand(n, n).t())
                dist_ap, dist_an = [], []
                for i in range(n):
                    dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
                    dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
                dist_ap = torch.cat(dist_ap)
                dist_an = torch.cat(dist_an)

                y = torch.ones_like(dist_an)
                loss = self.ranking_loss(dist_an, dist_ap, y)

            else:

                inputs = inputss[batch_size*(i + 1):]
                targets = targetss[batch_size*(i + 1):]

                n = inputs.size(0)

                dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
                dist = dist + dist.t()
                dist.addmm_(1, -2, inputs, inputs.t())
                dist = dist.clamp(min=1e-12).sqrt()

                mask = targets.expand(n, n).eq(targets.expand(n, n).t())
                dist_ap, dist_an = [], []
                for i in range(n):
                    dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
                    dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
                dist_ap = torch.cat(dist_ap)
                dist_an = torch.cat(dist_an)

                y = torch.ones_like(dist_an)
                loss = self.ranking_loss(dist_an, dist_ap, y)

            losss += loss

        return losss

class CosineLoss(nn.Module):

    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, inputs, targets):

        # targets: (m*n)
        # cluster_center: (m*n)

        m = len(inputs[:,0])
        similarity = torch.cosine_similarity(inputs, targets, dim=1)
        similarity_sum = torch.sum(similarity)
        loss = 1 - similarity_sum/m

        return loss

