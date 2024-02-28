import torch
from torch.utils.data import DataLoader, TensorDataset
from botorch.acquisition.analytic import AnalyticAcquisitionFunction


class TSAcquisitionFunction(AnalyticAcquisitionFunction):
    """
    Thompson sampling acquisition function. While it uses a posterior sample, it is an analytic one.
    I.e. once we pick a sample of the posterior f_s ~ p(f | D), f_s is a deterministic function over x.

    Parameters:
    -----------
    model: botorch.models.model.Model

    posterior_transform: botorch.acquisition.objective.PosteriorTransform
        Optional

    maximize: bool, default = True
        Whether to maximize the acqf f_s or minimize it

    random_state: int, default = 123
        The random state of the sampling f_s ~ p(f | D). This is to ensure that for any given x,
        the sample from p(f(x) | D) comes from the same sample posterior sample f_s ~ p(f | D).
    """
    def __init__(self, model, posterior_transform=None, maximize=True, random_state=123):
        super().__init__(model, posterior_transform)
        self.maximize = maximize
        self.random_state = random_state

    def forward(self, x):
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
        generator = torch.Generator(device=x.device).manual_seed(self.random_state)
        eps = torch.randn(*std.shape, device=x.device, generator=generator)
        f_sample = mean + std * eps

        # BoTorch assumes acqf to be maximization
        # https://github.com/pytorch/botorch/blob/0e74bb60be3492590ea88d6373d89a877c6a52c1/botorch/generation/gen.py#L249-L252
        return f_sample if self.maximize else -f_sample


def thompson_sampling_with_cand(model, x_cand, maximization=True, batch_size=128, random_state=123):
    """
    Thompson sampling for BoTorch on discrete candidates from the input space.
    Supports single objective only.

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

    random_state: int
        The same random state used throughout the batches

    Returns:
    --------
    x_best: torch.Tensor
        Shape (dim,). The argmax/argmin of the posterior samples.
    """
    dataloader = DataLoader(TensorDataset(x_cand), batch_size=batch_size)
    generator = torch.Generator(device=x_cand.device).manual_seed(random_state)

    best_x, best_fx = None, (-torch.inf if maximization else torch.inf)
    with torch.no_grad():
        for (x,) in dataloader:
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # Add the BoTorch's q-dimension

            posterior = model.posterior(x)
            f_mean, f_var = posterior.mean, posterior.variance
            eps = torch.randn(*f_var.shape, device=x_cand.device, generator=generator)
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

    return best_x
