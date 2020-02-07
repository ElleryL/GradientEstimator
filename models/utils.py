import math

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F



def gaussian_log_prob(mu, std, x):

    if len(mu.shape) != len(x.shape):
        mu = mu.unsqueeze(0).repeat(x.shape[0],1,1)
        std = std.unsqueeze(0).repeat(x.shape[0], 1, 1)
    return -.5 * (x - mu).pow(2) / std.pow(2) - (std).log() - .5 * np.log(2 * np.pi)

def bernoulli_log_prob(param,data):
    param = param.clamp(1e-6,1-1e-6)
    return data * param.log() + (1-data) * (1-param).log()