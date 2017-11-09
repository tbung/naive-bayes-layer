import torch
import torch.nn as nn
from torch.autograd import Variable

from numpy import pi
import numpy as np


class GaussianNaiveBayes(nn.Module):
    """ Implementation of Naive Bayes as a layer for pytorch models

    TODO
    ----
    - Make std devs fixable
    - Look into better param initialization
    """
    def __init__(self, features, classes, fix_variance=True):
        super(self.__class__, self).__init__()

        self.features = features
        self.classes = classes

        # We need mean and variance per feature and class
        self.register_parameter(
            "means",
            nn.Parameter(torch.Tensor(self.classes, self.features))
        )
        if not fix_variance:
            self.register_parameter(
                "variances",
                nn.Parameter(torch.Tensor(self.classes, self.features))
            )
        else:
            self.register_buffer(
                "variances",
                Variable(torch.Tensor(self.classes, self.features))
            )

        # We need the class priors
        self.register_parameter(
            "class_priors",
            nn.Parameter(torch.Tensor(self.classes))
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.means.data.normal_()
        self.variances.data.fill_(1)
        self.class_priors.data.fill_(1/self.classes)

    def forward(self, x):
        x = x[:,np.newaxis,:]
        return (torch.log(self.class_priors)
                + torch.sum(- 0.5 * torch.log(2 * pi * torch.abs(self.variances)), dim=1)
                - torch.sum((x - self.means)**2 / torch.abs(self.variances) / 2, dim=1))
