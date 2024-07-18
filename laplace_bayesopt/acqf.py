from __future__ import annotations
from botorch.acquisition.objective import PosteriorTransform
import torch
from torch.utils.data import DataLoader, TensorDataset
from botorch.acquisition.analytic import AnalyticAcquisitionFunction


class IndependentThompsonSampling(AnalyticAcquisitionFunction):
    """
    While it uses a posterior sample, it is an analytic one.
    I.e. once we pick a sample of the posterior f_s ~ p(f | D), f_s is a deterministic function over x.
    Note that, for the computational tractability, we sample from the diagonal of the
    GP posterior, i.e. sample f(x) ~ N(mu(x), sigma^2(x)) independently for each x.
    This ignores e.g. the smoothness of the kernel, but still a valid acquisition function
    nonetheless. It's akin to the posterior-mean acq. func. but with noise coming from
    the posterior variance.

    Parameters:
    -----------
    model: botorch.models.model.Model

    posterior_transform: botorch.acquisition.objective.PosteriorTransform
        Optional

    maximize: bool, default = True
        Whether to maximize the acqf f_s or minimize it

    random_state: int or None, default = None
        If set, then the stochasticity is only present within batch.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        posterior_transform: PosteriorTransform | None = None,
        maximize: bool = True,
        random_state: int | None = None,
    ):
        super().__init__(model, posterior_transform)
        self.maximize = maximize
        self.random_state = random_state

    def forward(self, x: torch.Tensor):
        """
        Parameters:
        -----------
        x: torch.Tensor
            Shape (n, 1, d)

        Returns:
        --------
        f_sample: torch.Tensor
            Shape (n,)
        """
        mean, std = self._mean_and_sigma(x)

        if len(mean.shape) == 0:
            mean = mean.unsqueeze(0)
        if len(std.shape) == 0:
            std = std.unsqueeze(0)

        if self.random_state is not None:
            generator = torch.Generator(device=x.device).manual_seed(self.random_state)
        else:
            generator = None

        eps = torch.randn(*mean.shape, device=x.device, generator=generator)
        f_sample = mean + std * eps

        # BoTorch assumes acqf to be maximization
        # https://github.com/pytorch/botorch/blob/0e74bb60be3492590ea88d6373d89a877c6a52c1/botorch/generation/gen.py#L249-L252
        return f_sample if self.maximize else -f_sample


def discrete_independent_thompson_sampling(
    model: torch.nn.Module,
    x_cand: torch.Tensor,
    maximization: bool = True,
    batch_size: int = 128,
    random_state: int | None = None,
):
    """
    Thompson sampling for BoTorch on discrete candidates from the input space.
    Supports single objective only. Note that, for the computational tractability, we
    sample from the diagonal of the GP posterior, i.e. sample f(x) ~ N(mu(x), sigma^2(x))
    independently for each x. This ignores e.g. the smoothness of the kernel, but
    still a valid acquisition function nonetheless. It's akin to the posterior-mean
    acq. func. but with noise coming from the posterior variance.

    Parameters:
    -----------
    model: botorch.models.model.Model
        Has method `posterior(x_cand)`, returning Gaussian over f(x_cand)

    x_cand: torch.Tensor
        Shape (num_candidates, dim)

    maximization: bool
        Whether to take max or min

    batch_size: int
        Batch size for each chunk of x_cand. Useful when num_candidates is large

    random_state: int or None
        The same random state used throughout the batches. If set, the stochasticity
        is only within batch.

    Returns:
    --------
    x_best: torch.Tensor
        Shape (dim,). The argmax/argmin of the posterior samples.

    fx_best: float
        The acquisition function value of x_best.
    """
    dataloader = DataLoader(TensorDataset(x_cand), batch_size=batch_size)

    if random_state is not None:
        generator = torch.Generator(device=x_cand.device).manual_seed(random_state)
    else:
        generator = None

    best_x, best_fx = None, (-torch.inf if maximization else torch.inf)
    with torch.no_grad():
        for (x,) in dataloader:
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # Add the BoTorch's q-dimension

            posterior = model.posterior(x)
            f_mean, f_var = posterior.mean, posterior.variance
            eps = torch.randn(*f_mean.shape, device=x_cand.device, generator=generator)
            f_sample = (f_mean + f_var.sqrt() * eps).flatten()

            if maximization:
                curr_max = torch.max(f_sample)
                if curr_max > best_fx:
                    best_fx = curr_max
                    best_x = x_cand[torch.argmax(f_sample)]
            else:
                curr_min = torch.min(f_sample)
                if curr_min < best_fx:
                    best_fx = curr_min
                    best_x = x_cand[torch.argmin(f_sample)]

    return best_x, best_fx
