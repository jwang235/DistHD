from dataclasses import dataclass
# from tkinter.tix import X_REGION
import Config
import torch
import sys
import random
import numpy as np
import sklearn
from Config import config, Update_T
from torch.nn.functional import normalize
import math

class HD_classifier:
    def __init__(self, param):
        self.D = param["D"]
        self.nClasses = param["nClasses"]
        self.classes = torch.zeros((self.nClasses, self.D))
        self.counts = torch.zeros(self.nClasses)
        self.param = param
        self.use_cuda = False
        self.distance1 = []
        self.distance2 = []
        
    def scores(self, data):
        data_normed = torch.nn.functional.normalize(data, p=2.0, dim=1, eps=1e-12, out=None)
        model_normed = torch.nn.functional.normalize(self.classes, p=2.0, dim=1, eps=1e-12, out=None)
        cdist = data_normed @ model_normed.T
        return cdist


    def train(self, data, label, param,  epochs = 20, batch = 1024): # From OnlineHD Iterative Fit
        lr = param["lr"]
        for epoch in range(epochs):
            for i in range(0, data.size(0), batch):
                data_ = data[i : i + batch] 
                label_ = label[i : i + batch]
                scores = self.scores(data_)
                y_pred = scores.argmax(1)
                wrong = label_ != y_pred
                aranged = torch.arange(data_.size(0))
                alpha1 = (1.0 - scores[aranged,label_]).unsqueeze_(1)
                alpha2 = (scores[aranged,y_pred] - 1.0).unsqueeze_(1)
                for lbl in label_.unique():
                    m1 = wrong & (label_ == lbl) # mask of missed true lbl
                    m2 = wrong & (y_pred == lbl) # mask of wrong preds
                    self.classes[lbl] += lr*(alpha1[m1]*data_[m1]).sum(0)
                    self.classes[lbl] += lr*(alpha2[m2]*data_[m2]).sum(0)
            

    def distance(self,data,label, param):
        model = torch.nn.functional.normalize(self.classes, p=2.0, dim=1, eps=1e-12, out=None)
        alpha, beta, theta = param["alpha"], param["beta"], param["theta"]
        scores = self.scores(data)
        pred1, pred2 = scores.topk(2)[1][ : , 0], scores.topk(2)[1][ : , 1]
        wrong = pred1 != label
        data_, pred2_, label_, pred1_ = data[wrong], pred2[wrong], label[wrong], pred1[wrong]
        
        partial = pred2_ == label_
        dist2corr = torch.abs(model[label_[partial]] - data_[partial])
        dist2incorr = torch.abs(model[pred1_[partial]] - data_[partial])
        partial_dist = torch.sum((beta * dist2incorr - alpha * dist2corr), 0) # partial correct
        
        complete = pred2_ != label_
        dist2corr = torch.abs(model[label_[complete]] - data_[complete])
        dist2incorr1 = torch.abs(model[pred1_[complete]] - data_[complete])
        dist2incorr2 = torch.abs(model[pred2_[complete]] - data_[complete])
        complete_dist = torch.sum((beta * dist2incorr1 + theta * dist2incorr2 - alpha * dist2corr), 0)   # completely incorrect
        return partial_dist, complete_dist
    

    def compare(self, data, label):
        scores = self.scores(data)
        pred =  (scores.topk(3)[1])
        top2_acc = ((label==pred[ : , 0]).sum() + (label==pred[ : , 1]).sum())/label.shape[0]
        top3_acc = ((label==pred[ : , 0]).sum() + (label==pred[ : , 1]).sum() + (label==pred[ : , 2]).sum())/label.shape[0]
        return top2_acc.item(), top3_acc.item()


    def test(self, data, label):
        scores = self.scores(data)
        pred = scores.argmax(1)
        accuracy = ((pred == label).sum() / (label.shape[0])).item()
        return accuracy
        
    def eval_model(self, dist1, dist2, param):
        regen = math.ceil(param["regen_rate"] * self.D)
        dist = dist2 + 0.5 * dist1
        toChange = torch.argsort(dist)[0:regen]
        return toChange
 
    def update_model(self, toChange = None):
        print("regenerate amount: ", len(toChange))
        self.classes[ : , toChange] = torch.zeros(self.nClasses, len(toChange))
       