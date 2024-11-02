"""
Following https://botorch.org/tutorials/composite_mtbo
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import tqdm
from botorch.acquisition import GenericMCObjective
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim.optimize import optimize_acqf
from botorch.sampling import IIDNormalSampler
from botorch.test_functions import Hartmann
from torch import nn

from laplace_bayesopt.botorch import LaplaceBoTorch

np.random.seed(10)
torch.set_default_dtype(torch.float64)
torch.manual_seed(10)


class ContextualHartmann6(Hartmann):
    def __init__(self, num_tasks: int = 3, noise_std=None, negate=False):
        super().__init__(dim=6, noise_std=noise_std, negate=negate)
        self.task_range = torch.linspace(0, 1, num_tasks).unsqueeze(-1)
        self._bounds = [(0.0, 1.0) for _ in range(self.dim - 1)]
        self.bounds = torch.tensor(self._bounds).t()

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        batch_X = X.unsqueeze(-2)
        batch_dims = X.ndim - 1

        expanded_task_range = self.task_range
        for _ in range(batch_dims):
            expanded_task_range = expanded_task_range.unsqueeze(0)
        task_range = expanded_task_range.repeat(*X.shape[:-1], 1, 1).to(X)
        concatenated_X = torch.cat(
            (
                batch_X.repeat(*[1] * batch_dims, self.task_range.shape[0], 1),
                task_range,
            ),
            dim=-1,
        )
        return super().evaluate_true(concatenated_X)


NUM_TASKS = 3
problem = ContextualHartmann6(num_tasks=NUM_TASKS, noise_std=0.001, negate=True)
weights = torch.randn(NUM_TASKS)


def callable_func(samples, X=None):
    res = -torch.cos((samples**2) + samples * weights)
    return res.squeeze().sum(dim=-1)


objective = GenericMCObjective(callable_func)
bounds = problem.bounds

n_init = 5
train_x = (bounds[1] - bounds[0]) * torch.rand(n_init, bounds.shape[1]) + bounds[0]
train_y = problem(train_x)


def get_net():
    return torch.nn.Sequential(
        nn.Linear(train_x.shape[-1], 20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.ReLU(),
        nn.Linear(20, NUM_TASKS),
    )


model = LaplaceBoTorch(get_net, train_x, train_y)

best_objective = objective(train_y).max()
pbar = tqdm.trange(100)
pbar.set_description(f"[Best objective = {best_objective:.3f}]")

# For qEI
NUM_SAMPLES = 4

for i in pbar:
    sampler = IIDNormalSampler(sample_shape=torch.Size([NUM_SAMPLES]))
    acq_f = qLogExpectedImprovement(
        model, best_f=best_objective, sampler=sampler, objective=objective
    )

    # Get a proposal for new x
    new_x, val = optimize_acqf(
        acq_f,
        bounds=bounds,
        q=4,
        num_restarts=11,
        raw_samples=22,
    )

    if len(new_x.shape) == 1:
        new_x = new_x.unsqueeze(0)

    # Evaluate the objectives of all tasks on the proposed x
    new_y = problem(new_x)  # (q, NUM_TASKS)

    # Update posterior
    model = model.condition_on_observations(new_x, new_y)

    # Evaluate the summarized objective (a scalar)
    curr_objective = objective(new_y).max()
    best_objective = objective(model.train_Y).max()
    pbar.set_description(
        f"[Best objective = {best_objective:.3f}, curr objective = {curr_objective:.3f}]"
    )
