import torch
import torch.distributions
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
elif not torch.cuda.is_available():
    FloatTensor = torch.FloatTensor

import torch.nn.functional as F

# Manually set the parameters
logging = 0
num_samples = 1000
num_inducing_points = 6
lr_kernel = 0.01
lr_ip = 0.1
num_epochs = 100


class GP_InducingPoints(torch.nn.Module):
    def __init__(self, _x=None, _y=None, _num_inducing_points=num_inducing_points, _dim=1):
        super().__init__()

        assert type(_x) != type(None) # some sanity checking
        assert type(_y) != type(None) # some sanity checking for the correct input

        self.x = _x # save data set for convenience sake, not recommended for large data sets
        self.y = _y

        self.num_inducing_points = _num_inducing_points

        inducing_x = torch.linspace(_x.min().item(), _x.max().item(), self.num_inducing_points).reshape(-1,1) # distribute the data points as a linspace between x.min() and x.max() to get a good initializaiton of the inducing points
        self.inducing_x_mu = torch.nn.Parameter(inducing_x + torch.randn_like(inducing_x).clamp(-0.1,0.1)) # add some noise to the x values of the inducing points
        self.inducing_y_mu = torch.nn.Parameter(FloatTensor(_num_inducing_points, _dim).uniform_(-0.5,0.5)) # since we normalized the data to N(0,1) we initialize the y values in the middle of N(0,1)

        self.length_scale = torch.nn.Parameter(torch.scalar_tensor(0.00001)) # the kernel hyperparameter to be optimized alongside inducing points
        self.noise = torch.nn.Parameter(torch.scalar_tensor(0.5)) # the noise hyperparameter to be optimized alongside inducing points

    def compute_kernel_matrix(self, x1, x2):
        assert x1.shape[1] == x2.shape[1] # check dimension
        assert x1.numel() >= 0 # sanity check
        assert x2.numel() >= 0 # sanity check

        pdist = (x1 - x2.T)**2 # outer difference
        # kernel_matrix = torch.exp(-0.5*1/(self.length_scale+0.001)*pdist)

        # use rational quadratic kernel
        kernel_matrix = (1 + pdist/(2*self.length_scale**2))**(-1)

        return kernel_matrix

    def forward(self, _X):
        # compute all the kernel matrices
        self.K_XsX = self.compute_kernel_matrix(_X, self.inducing_x_mu)
        self.K_XX = self.compute_kernel_matrix(self.inducing_x_mu, self.inducing_x_mu)
        self.K_XsXs = self.compute_kernel_matrix(_X, _X)

        # invert K_XX and regularizing it for numerical stability
        self.K_XX_inv = torch.inverse(self.K_XX + 1e-10*torch.eye(self.K_XX.shape[0]))

        # compute mean and covariance for forward prediction
        mu = self.K_XsX @ self.K_XX_inv @ self.inducing_y_mu
        sigma = self.K_XsXs - self.K_XsX @ self.K_XX_inv @ self.K_XsX.T + self.noise*torch.eye(self.K_XsXs.shape[0])

        # for each point in _X output MAP estimate and variance of prediction ( that's the torch.diag (...) )
        return mu, torch.diag(sigma)[:, None]

    def NMLL(self, _X, _y):
        # set reasonable constraints on the optimizable parameters
        self.length_scale.data = self.length_scale.data.clamp(0.00001, 3.0)
        self.noise.data = self.noise.data.clamp(0.000001, 3)

        # compute all the kernel matrices again ... now you see why we want to use inducing points
        K_XsXs = self.compute_kernel_matrix(_X, _X)
        K_XsX = self.compute_kernel_matrix(_X, self.inducing_x_mu)
        K_XX = self.compute_kernel_matrix(self.inducing_x_mu, self.inducing_x_mu)
        K_XX_inv = torch.inverse(K_XX + 1e-10*torch.eye(K_XX.shape[0]))

        Q_XX = K_XsXs - K_XsX @ K_XX_inv @ K_XsX.T

        # compute mean and covariance and GP distribution itself
        mu = K_XsX @ K_XX_inv @ self.inducing_y_mu
        Sigma = Q_XX + self.noise**2*torch.eye(Q_XX.shape[0]) # noise regularized covariance

        # Use Cholesky decomposition for stability
        L = torch.cholesky(Sigma, upper=False)
        p_y = MultivariateNormal(mu.squeeze(), scale_tril=L)
        mll = p_y.log_prob(_y.squeeze()) # evaluate the probability of the target values in the training data set under the distribution of the GP

        mll -= 1/(2 * self.noise**2) * torch.trace(Q_XX) # add a regularization term to regularize variance

        return -mll

    def plot(self, _title=""):
        x = torch.linspace(self.x.min()*1.5, self.x.max()*1.5, 200).reshape(-1,1)

        with torch.no_grad():
            mu, sigma = self.forward(x)

        x = x.numpy().squeeze()
        mu = mu.numpy().squeeze()
        sigma = sigma.numpy().squeeze()

        plt.title(_title)
        plt.scatter(self.inducing_x_mu.detach().numpy(), self.inducing_y_mu.detach().numpy(), c='k')
        plt.scatter(self.x.detach().numpy(), self.y.detach().numpy(), c='r', alpha=0.1)
        plt.fill_between(x, mu-3*sigma, mu+3*sigma, alpha=0.1, color='blue')
        plt.plot(x, mu)
        # plt.xlim(self.x.min()*1.5, self.x.max()*1.5)
        # plt.ylim(-3, 3)
        plt.show()

# Load CSV data
data = pd.read_csv("CH4_CuBTC_Test.csv")
X = data['X1'].values.reshape(-1, 1)
X =  np.log(X)
y = data['y'].values.reshape(-1, 1)
y =  np.log(y)

# Normalize the data using the mean and standard deviation from the training set
x_m = X.mean()
x_std = X.std()

y_m = y.mean()
y_std = y.std()

X_scaled = (X - x_m) / x_std
y_scaled = (y - y_m) / y_std

X = FloatTensor(X_scaled)
y = FloatTensor(y_scaled)

# Initialize the GP and plot initial prediction
gp = GP_InducingPoints(_x=X, _y=y)
gp.plot(_title="Init")

# Use two different learning rates since inducing points need to potentially cover a far larger distance than kernel parameters
# optim = torch.optim.Adam([{"params": [gp.length_scale, gp.noise], "lr": lr_kernel},
#                           {"params": [gp.inducing_x_mu, gp.inducing_y_mu], "lr": lr_ip}])

# optimizing the kernel parameters and inducing points together
# optim = torch.optim.Adam([gp.length_scale, gp.noise, gp.inducing_x_mu, gp.inducing_y_mu], lr=lr_kernel)

# use a RMS prop optimizer
optim = torch.optim.RMSprop([gp.length_scale, gp.noise, gp.inducing_x_mu, gp.inducing_y_mu], lr=lr_kernel)

# Put it all in a data loader and let it train
train_loader = DataLoader(TensorDataset(FloatTensor(X), FloatTensor(y)),
                        batch_size=num_samples,
                        shuffle=True,
                        num_workers=1)

for epoch in range(num_epochs):
    for i, (data, label) in enumerate(train_loader):
        optim.zero_grad()
        mll = gp.NMLL(data, label)
        mll.backward()
        optim.step()

        if epoch % (num_epochs // 10) == 0:
            gp.plot(_title=f"Training Epoch {epoch:.0f}")
        
            # print the optim parameters as training progresses
            print(gp.length_scale)
            print(gp.noise)
gp.plot(_title="Post Training")



def predict():
    final_inducing_points_x_original_scale = gp.inducing_x_mu.detach().cpu().numpy() * x_std + x_m
    final_inducing_points_y_original_scale = gp.inducing_y_mu.detach().cpu().numpy() * y_std + y_m

    final_inducing_points_x_original_scale = np.exp(final_inducing_points_x_original_scale)
    final_inducing_points_y_original_scale = np.exp(final_inducing_points_y_original_scale)

    # # plot the final inducing points but first take X and y back to the original scale

    X_original_scale = X * x_std + x_m
    y_original_scale = y * y_std + y_m

    X_original_scale = np.exp(X_original_scale)
    y_original_scale = np.exp(y_original_scale)

    plt.figure()
    plt.scatter(X_original_scale, y_original_scale, alpha=0.1, c='r')
    plt.scatter(final_inducing_points_x_original_scale, final_inducing_points_y_original_scale, c='k')


    # now use the final inducing points to make predictions of the original scale X
    x = X_original_scale
    x = np.log(x)
    x_scaled = (x - x_m) / x_std
    x_scaled = FloatTensor(x_scaled)

    with torch.no_grad():
        mu, sigma = gp.forward(x_scaled)

    mu = mu.detach().cpu().numpy() * y_std + y_m
    sigma = sigma.detach().cpu().numpy() * y_std + y_m

    mu = np.exp(mu)
    print(mu)

    plt.figure()
    plt.plot(X_original_scale, mu, alpha=0.1, c='r')
    # plt.scatter(final_inducing_points_x_original_scale, final_inducing_points_y_original_scale, c='k')
    # plt.fill_between(x.squeeze(), mu.squeeze()-3*sigma.squeeze(), mu.squeeze()+3*sigma.squeeze(), alpha=0.1, color='blue')
    plt.scatter(X_original_scale, y_original_scale)

predict()
