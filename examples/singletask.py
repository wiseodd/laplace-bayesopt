from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")


import numpy as np
import torch
import tqdm
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim.optimize import optimize_acqf
from botorch.test_functions import Branin
from torch import distributions as dists
from torch import nn
from torch.nn import functional as F

from laplace_bayesopt.botorch import LaplaceBoTorch

np.random.seed(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(1)

true_f = Branin()
bounds = torch.tensor([[-5, 10], [0, 15]]).T.double()

train_data_points = 20

train_x = torch.cat(
    [
        dists.Uniform(*bounds.T[i]).sample((train_data_points, 1))
        for i in range(2)  # for each dimension
    ],
    dim=1,
)
train_y = true_f(train_x).reshape(-1, 1)

test_x = torch.cat(
    [
        dists.Uniform(*bounds.T[i]).sample((10000, 1))
        for i in range(2)  # for each dimension
    ],
    dim=1,
)
test_y = true_f(test_x)


def get_net():
    return torch.nn.Sequential(
        nn.Linear(2, 50), nn.ReLU(), nn.Linear(50, 50), nn.ReLU(), nn.Linear(50, 1)
    )


model = LaplaceBoTorch(get_net, train_x, train_y)


def evaluate_model(model):
    pred, _ = model._get_prediction(test_x, use_test_loader=True, joint=False)
    return F.mse_loss(pred, test_y).squeeze().item()


best_y = train_y.min().item()
trace_best_y = []
pbar = tqdm.trange(100)
pbar.set_description(f"[MSE = {evaluate_model(model):.3f}; Best f(x) = {best_y:.3f}]")

for i in pbar:
    acq_f = ExpectedImprovement(model, best_f=best_y, maximize=False)

    # Get a proposal for new x
    new_x, val = optimize_acqf(
        acq_f,
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=20,
    )

    if len(new_x.shape) == 1:
        new_x = new_x.unsqueeze(0)

    # Evaluate the objective on the proposed x
    new_y = true_f(new_x).unsqueeze(-1)  # (q, 1)

    # Update posterior
    model = model.condition_on_observations(new_x, new_y)

    # Update the current best y
    if new_y.min().item() <= best_y:
        best_y = new_y.min().item()

    trace_best_y.append(best_y)
    pbar.set_description(
        f"[MSE = {evaluate_model(model):.3f}; Best f(x) = {best_y:.3f}, curr f(x) = {new_y.min().item():.3f}]"
    )
