import mindspore
import numpy as np
from mindspore import nn
from mindspore.common.initializer import Normal
from mindspore.ops import operations as P
from mindspore import Tensor
import pandas as pd
import random
import mindspore.dataset.transforms as transforms
from mindspore import Model
import os
import dgckernel
from math import sin,cos,asin,sqrt,radians
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import mindspore.ops.functional as F
from tqdm import tqdm

class Actor(nn.Cell):
    """Lenet network structure."""
    # define the operator required

    def __init__(self):
        super(Actor, self).__init__(auto_prefix=True)
        self.embeding_grid=nn.Embedding(vocab_size=331,embedding_size=7)

        self.embeding_timestamp=nn.Embedding(vocab_size=288,embedding_size=7)

        self.concat = P.Concat(axis=1)
        self.dense1 = nn.Dense(in_channels=21,out_channels=32)
        self.relu=nn.ReLU()
        self.dense2 = nn.Dense(in_channels=32,out_channels=64)
        self.dropout=nn.Dropout(0.6)
        self.dense3=nn.Dense(in_channels=64,out_channels=7)
        self.softmax=nn.Softmax()
    # use the preceding operators to construct networks
    def construct(self,all):

        x1=self.embeding_grid(all[0])

        x2=self.embeding_timestamp(all[1])
        x3=all[2]
        x=self.concat((x3,x1,x2))
        x=self.dense1(self.relu(x))
        x=self.dense2(self.relu(x))
        x=self.dropout(x)

        x=self.dense3(x)
        x=self.softmax(x)
        return x