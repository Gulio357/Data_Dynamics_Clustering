import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import torch
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

from sklearn.neighbors import KNeighborsClassifier



def system(v, t):
    v1, v2, v3 = v
    dv1dt = v1 * v3
    dv2dt = -v2 * v3
    dv3dt = -v1**2 + v2**2
    return [dv1dt, dv2dt, dv3dt]


def sample_point():
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    return x, y


def generate_solutions(initial_settings):
    t = np.linspace(0, 30, 10000)

    dataset = []

    for i in range(len(initial_settings)):

        v0 = initial_settings[i]
        v0 = [1.0, 0.1*v0[0], 1.0*v0[1]]
        v = odeint(system, v0, t)

        dataset.append(v[:, 0])
        
    dataset = np.array(dataset)

    return dataset

class MultiOutputGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_components):
        super(MultiOutputGPModel, self).__init__(train_x, train_y, likelihood)
        
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([n_components]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([n_components])),
            batch_shape=torch.Size([n_components])
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )