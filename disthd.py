import Config
import encoder
import basis
import classifier
import data_preprocess as pre

import numpy as np
import sklearn.datasets
import sklearn.preprocessing
import sklearn.model_selection
import math 
import copy
import torch


class DistHD(object):
    def __init__(self, D, n_features, n_classes, lr):
        self.param =  Config.config
        self.param["D"] = D
        self.param["nFeatures"] = n_features
        self.param["nClasses"] = n_classes
        self.param["lr"] = lr
        self.hdb = basis.HD_basis(basis.Generator.Vanilla, self.param)
        self.hde = encoder.HD_encoder(self.hdb.getBasis(), self.param)
        self.hdc = classifier.HD_classifier(self.param)
        self.accs = []
        self.best_hdc = []
        self.best_hdb = []
    
    def train(self, x_train, y_train, epochs, nRegen):
        train = self.hde.encodeData(x_train, self.param)
        for i in range(nRegen):
            self.hdc.train(train, y_train, self.param, epochs)
            acc = self.hdc.test(train, y_train)
            self.accs.append(acc)
            if acc == max(self.accs):
                self.best_hdc, self.best_hdb = copy.deepcopy(self.hdc), copy.deepcopy(self.hdb)
                print("better acc achieved! ", acc)
            dist1, dist2 = self.hdc.distance(train, y_train, self.param)
            toChange = self.hdc.eval_model(dist1, dist2, self.param)
            self.hdc.update_model(toChange)
            self.hdb.update_basis(toChange)
            self.hde = encoder.HD_encoder(self.hdb.getBasis(), self.param)
            train = self.hde.encodeData(x_train, self.param)
            print("completed epoch ", i, ", the acc is ", acc)
    
    def test(self, x_train, y_train, x_test, y_test):
        self.hde = encoder.HD_encoder(self.best_hdb.getBasis(), self.param)
        train = self.hde.encodeData(x_train, self.param)
        acc1 = self.best_hdc.test(train, y_train)
        test = self.hde.encodeData(x_test, self.param)
        acc2 = self.best_hdc.test(test, y_test)
        print(acc1, acc2)
        
    def compare(self, X, y, epochs):
        x_train, x_test = self.hde.encodeData(X[0], self.param), self.hde.encodeData(X[1], self.param)
        self.hdc.train(x_train, y[0], self.param, epochs)
        top1 = self.hdc.test(x_test, y[1])
        top2, top3 = self.hdc.compare(x_test, y[1])
        return top1, top2, top3

if __name__ == "__main__":
    # X, Y, n_classes, n_features = pre.mnist() #mnist
    X, Y, n_classes, n_features = pre.isolet() #isolet
    #X, Y, n_classes, n_features = pre.ucihar() # UCIHAR
    model = DistHD(D = 5000, n_features = n_features, n_classes=n_classes, lr=0.35)
    model.train(X[0], Y[0], epochs = 20, nRegen = 20)
    model.test(X[0], Y[0], X[1], Y[1])
    compare = DistHD(D = 20000, n_features = n_features, n_classes = n_classes, lr = 0.35)
    baseline = compare.compare(X, Y, epochs = 10)
    print(baseline)
    

