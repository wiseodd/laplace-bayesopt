"""
Reference: https://botorch.org/tutorials/multi_objective_bo
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")


import numpy as np
import torch
import tqdm
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from torch import nn

from laplace_bayesopt.botorch import LaplaceBoTorch

np.random.seed(10)
torch.manual_seed(10)
torch.set_default_dtype(torch.float64)

NOISE_SE = torch.tensor([15.19, 0.63])
BATCH_SIZE = 4
NUM_RESTARTS = 10
RAW_SAMPLES = 20

problem = BraninCurrin(negate=True)


def get_net():
    return torch.nn.Sequential(
        nn.Linear(problem.dim, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, problem.num_objectives),
    )


def generate_initial_data(n=6):
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n, q=1).squeeze(1)
    train_y_true = problem(train_x)
    train_y = train_y_true + torch.randn_like(train_y_true) * NOISE_SE
    return train_x, train_y, train_y_true


def initialize_model(train_x, train_y):
    train_x = normalize(train_x, problem.bounds)
    return LaplaceBoTorch(
        get_net,
        train_x,
        train_y,
        outcome_transform=Standardize(m=problem.num_objectives),
    )


standard_bounds = torch.zeros(2, problem.dim)
standard_bounds[1] = 1

train_x, train_y, train_y_true = generate_initial_data(n=2 * (problem.dim + 1))
model = initialize_model(train_x, train_y)

# Compute hypervolume
bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_y_true)
hypervolume = bd.compute_hypervolume().item()


def log_hypervolume_difference(hypervolume):
    return np.log10(problem.max_hv - hypervolume)


pbar = tqdm.trange(100)
pbar.set_description(
    f"[Log hypervolume diff = {log_hypervolume_difference(hypervolume):.3f}]"
)

for i in pbar:
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

    with torch.no_grad():
        pred, _ = model._get_prediction(
            normalize(model.train_X, problem.bounds),
            # !Important!, so that pred has shape (batch, n_outputs)
            # instead of (batch*n_outptus)
            joint=False,
            use_test_loader=True,
        )

    partitioning = FastNondominatedPartitioning(
        ref_point=problem.ref_point,
        Y=pred,
    )
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point,
        partitioning=partitioning,
        sampler=sampler,
    )

    # Optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )

    # Observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_y_true = problem(new_x)
    new_y = new_y_true + torch.randn_like(new_y_true) * NOISE_SE

    # Update observations
    train_x = torch.cat([train_x, new_x])
    train_y = torch.cat([train_y, new_y])
    train_y_true = torch.cat([train_y_true, new_y_true])

    # Update posterior
    model = model.condition_on_observations(new_x, new_y)

    # Track performance
    bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_y_true)
    hypervolume = bd.compute_hypervolume().item()

    pbar.set_description(
        f"[Log hypervolume diff = {log_hypervolume_difference(hypervolume):.3f}]"
    )
