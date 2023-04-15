import torch
import math
import numpy as np
from enum import Enum


class HD_encoder:
    def __init__(self, basis, param):
        self.basis = basis
        self.param = param
        self.base = torch.empty(param["D"]).uniform_(0.0, 2*math.pi)
     
    def encodeData(self, data, param):
        encoded_data = torch.empty(param["D"], param["nFeatures"])
        # print(data)
        encoded_data = torch.matmul(data, self.basis.T)# .cos_()
        return encoded_data

    # Update basis of the HDE
    def updateBasis(self, basis):
        self.basis = basis

