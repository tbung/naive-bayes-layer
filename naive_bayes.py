import torch
import torch.nn as nn

from numpy import pi


class GaussianNaiveBayes(nn.Module):
    """ Implementation of Naive Bayes as a layer for pytorch models

    TODO
    ----
    - Make std devs fixable
    
    """
    def __init__(self, features, classes):
        super(self.__class__, self).__init__()

        # We need mean and variance per feature and class
        self.register_parameter(
            "means",
            nn.Parameter(torch.Tensor(self.classes, self.features))
        )
        self.register_parameter(
            "variances",
            nn.Parameter(torch.Tensor(self.classes, self.features))
        )

        # We need the class priors
        self.register_parameter(
            "log_class_priors",
            nn.Parameter(torch.Tensor(self.classes))
        )

    def forward(self, x):
        return (self.log_class_priors
                + torch.sum(- 0.5 * torch.log(2 * pi * self.variances), axis=1)
                + torch.sum((x - self.means)**2 / self.variances, axis=1))
