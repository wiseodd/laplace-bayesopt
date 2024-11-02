from __future__ import annotations
import torch
from torch import nn, optim
import torch.utils.data as data_utils

from gpytorch import distributions as gdists

import botorch.models.model as botorch_model
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform

from laplace import BaseLaplace, Laplace
from laplace.curvature import CurvlinopsGGN, CurvatureInterface
from laplace.marglik_training import marglik_training

from typing import Optional, Callable, List, Type
import math


class LaplaceBoTorch(botorch_model.Model):
    """
    BoTorch surrogate model with a Laplace-approximated Bayesian
    neural network. The Laplace class is defined in the library
    laplace-torch; install via:
    `pip install https://github.com/aleximmer/laplace.git`.

    Args:
    -----
    get_net: function None -> nn.Module
        Function that doesn't take any args and return a PyTorch model.
        Example usage: `get_net=lambda: nn.Sequential(...)`.

    train_X : torch.Tensor
        Training inputs of size (n_data, ...).

    train_Y : torch.Tensor
        Training targets of size (n_data, n_tasks).

    input_transform : botorch.models.transforms.input.InputTransform, optional
        Optional transformation on X.

    outcome_transform : botorch.models.transforms.outcome.OutcomeTransform, optional
        Optional transformation on Y.

    bnn : Laplace, optional, default=None
        When creating a new model from scratch, leave this at None.
        Use this only to update this model with a new observation during BO.

    likelihood : {'regression', 'classification'}
        Indicates whether the problem is regression or classification.

    noise_var : float | None, default=None.
        Output noise variance. If float, must be >= 0. If None,
        it is learned by marginal likelihood automatically.

    last_layer : bool, default False
        Whether to do last-layer Laplace. If True, then the model used is the
        so-called "neural linear" model.

    hess_factorization : {'full', 'diag', 'kron'}, default='kron'
        Which Hessian factorization to use to do Laplace. 'kron' provides the best
        tradeoff between speed and approximation error.

    marglik_mode : {'posthoc', 'online'}, default='posthoc'
        Whether to do online marginal-likelihood training or do standard NN training
        and use marglik to optimize hyperparams post-hoc.

    posthoc_marglik_iters: int > 0, default=100
        Number of iterations of post-hoc marglik tuning.

    online_marglik_freq: int > 0 default=50
        How often (in terms of training epoch) to do online marglik tuning.

    batch_size : int, default=10
        Batch size to use for the NN training and doing Laplace.

    n_epochs : int, default=1000
        Number of epochs for training the NN.

    lr : float, default=1e-1
        Learning rate to use for training the NN.

    wd : float, default=1e-3
        Weight decay for training the NN.

    device : {'cpu', 'cuda'}, default='cpu'
        Which device to run the experiment on.

    enable_backprop: bool, default=True
        Whether to enable backprop through the functional posterior. Set this to false
        if the BO problem is discrete.
    """

    def __init__(
        self,
        get_net: Callable[[], nn.Module],
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        bnn: Optional[BaseLaplace] = None,
        likelihood: str = "regression",
        noise_var: Optional[float] = None,
        last_layer: bool = False,
        hess_factorization: str = "kron",
        marglik_mode: str = "posthoc",
        posthoc_marglik_iters: int = 100,
        online_marglik_freq: int = 50,
        batch_size: int = 10,
        n_epochs: int = 1000,
        lr: float = 1e-1,
        wd: float = 1e-3,
        backend: Type[CurvatureInterface] = CurvlinopsGGN,
        device: str = "cpu",
        enable_backprop: bool = True,
    ):
        super().__init__()

        self.orig_train_X = train_X
        self.orig_train_Y = train_Y

        self.train_X = self.transform_inputs(train_X)
        if input_transform is not None:
            self.input_transform = input_transform
            self.input_transform.eval()

        if outcome_transform is not None:
            transformed_Y, _ = outcome_transform(train_Y)
            self.train_Y = transformed_Y
            self.outcome_transform = outcome_transform
            self.outcome_transform.eval()
        else:
            self.train_Y = train_Y

        assert likelihood in ["regression"]  # For now
        self.likelihood = likelihood
        self.batch_size = batch_size
        self.last_layer = last_layer
        self.subset_of_weights = "last_layer" if last_layer else "all"
        self.hess_factorization = hess_factorization
        self.posthoc_marglik_iters = posthoc_marglik_iters
        self.online_marglik_freq = online_marglik_freq
        assert device in ["cpu", "cuda"]
        self.device = device
        assert marglik_mode in ["posthoc", "online"]
        self.marglik_mode = marglik_mode
        self.n_epochs = n_epochs
        self.lr = lr
        self.wd = wd
        self.backend = backend
        self.get_net = get_net
        self.net = get_net()  # Freshly initialized
        self.bnn = bnn
        self.enable_backprop = enable_backprop

        if noise_var is float and noise_var is not None:
            raise ValueError("Noise variance must be float >= 0. or None")
        if noise_var is float and noise_var < 0:
            raise ValueError("Noise variance must be >= 0.")
        self.noise_var = noise_var

        # Initialize Laplace
        if self.bnn is None:
            self._train_model(self._get_train_loader())

    def posterior(
        self,
        X: torch.Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform=None,
        **kwargs,
    ) -> Posterior:
        if self.bnn is None:
            raise ValueError("The surrogate has not been trained!")

        # Notation:
        # ---------
        # B is the batch size
        # Q is the num. of x's predicted jointly
        # D is the feature size
        # K is the output size, i.e. num of tasks
        if hasattr(self, "input_transform"):
            self.input_transform.eval()
            X = self.input_transform(X)

        assert len(X.shape) == 2 or len(X.shape) == 3, "X must be a 2- or 3-d tensor"
        if len(X.shape) == 2:
            X = X.unsqueeze(1)

        # Transform to `(B*Q, D)`
        B, Q, D = X.shape
        X = X.reshape(B * Q, D)

        # Posterior predictive distribution
        # mean_y is (B*Q, K); cov_y is (B*Q*K, B*Q*K)
        mean_y, cov_y = self._get_prediction(X, use_test_loader=False, joint=True)

        # Mean must be `(B, Q*K)`
        K = self.num_outputs
        mean_y = mean_y.reshape(B, Q * K)

        # Cov must be `(B, Q*K, Q*K)`
        cov_y += self.bnn.sigma_noise**2 * torch.eye(B * Q * K, device=self.device)
        cov_y = cov_y.reshape(B, Q, K, B, Q, K)
        cov_y = torch.einsum("bqkbrl->bqkrl", cov_y)  # (B, Q, K, Q, K)
        cov_y = cov_y.reshape(B, Q * K, Q * K)

        if K > 1:
            dist = gdists.MultitaskMultivariateNormal(
                mean_y.reshape(B, Q, K), covariance_matrix=cov_y
            )
        else:
            dist = gdists.MultivariateNormal(mean_y, covariance_matrix=cov_y)

        post_pred = GPyTorchPosterior(dist)

        if hasattr(self, "outcome_transform"):
            post_pred = self.outcome_transform.untransform_posterior(post_pred)

        if posterior_transform is not None:
            post_pred = posterior_transform(post_pred)

        return post_pred

    def condition_on_observations(
        self, X: torch.Tensor, Y: torch.Tensor, **kwargs
    ) -> LaplaceBoTorch:
        # Append new observation to the current untrasformed data
        train_X = torch.cat([self.orig_train_X, X], dim=0)
        train_Y = torch.cat([self.orig_train_Y, Y], dim=0)

        return LaplaceBoTorch(
            # Replace the dataset & retrained BNN
            get_net=self.get_net,
            train_X=train_X,  # Important!
            train_Y=train_Y,  # Important!
            input_transform=getattr(self, "input_transform", None),
            outcome_transform=getattr(self, "outcome_transform", None),
            bnn=None,  # Important! `None`` so that Laplace is retrained
            likelihood=self.likelihood,
            noise_var=self.noise_var,
            last_layer=self.last_layer,
            hess_factorization=self.hess_factorization,
            marglik_mode=self.marglik_mode,
            posthoc_marglik_iters=self.posthoc_marglik_iters,
            online_marglik_freq=self.online_marglik_freq,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            lr=self.lr,
            wd=self.wd,
            backend=self.backend,
            device=self.device,
        )

    def _get_prediction(self, test_X: torch.Tensor, joint=True, use_test_loader=False):
        """
        Batched Laplace prediction.

        Args:
        -----
        test_X: torch.Tensor
            Array of size `(batch_shape, feature_dim)`.

        joint: bool, default=True
            Whether to do joint predictions (like in GP).

        use_test_loader: bool, default=False
            Set to True if your test_X is large.


        Returns:
        --------
        mean_y: torch.Tensor
            Tensor of size `(batch_shape, num_tasks)`.

        cov_y: torch.Tensor
            Tensor of size `(batch_shape*num_tasks, batch_shape*num_tasks)`
            if joint is True. Otherwise, `(batch_shape, num_tasks, num_tasks)`.
        """
        if self.bnn is None:
            raise Exception("Train your model first before making prediction!")

        if not use_test_loader:
            mean_y, cov_y = self.bnn(test_X.to(self.device), joint=joint)
        else:
            test_loader = data_utils.DataLoader(
                data_utils.TensorDataset(test_X, torch.zeros_like(test_X)),
                batch_size=256,
            )

            mean_y, cov_y = [], []

            for X_batch, _ in test_loader:
                X_batch = X_batch.to(self.device)
                _mean_y, _cov_y = self.bnn(X_batch, joint=joint)
                mean_y.append(_mean_y)
                cov_y.append(_cov_y)

            mean_y = torch.cat(mean_y, dim=0).squeeze()
            cov_y = torch.cat(cov_y, dim=0).squeeze()

        return mean_y, cov_y

    @property
    def num_outputs(self) -> int:
        """The number of outputs of the model."""
        return self.train_Y.shape[-1]

    def _train_model(self, train_loader):
        del self.bnn

        if self.marglik_mode == "posthoc":
            self._posthoc_laplace(train_loader)
        else:
            # Online
            la, model, _, _ = marglik_training(
                # Ensure that the base net is re-initialized
                self.net,
                train_loader,
                likelihood=self.likelihood,
                hessian_structure=self.hess_factorization,
                n_epochs=self.n_epochs,
                backend=self.backend,
                optimizer_kwargs={"lr": self.lr},
                scheduler_cls=optim.lr_scheduler.CosineAnnealingLR,
                scheduler_kwargs={"T_max": self.n_epochs * len(train_loader)},
                marglik_frequency=self.online_marglik_freq,
                enable_backprop=self.enable_backprop,  # Important!
            )
            self.bnn = la

        # Override sigma_noise if self.noise_var is not None
        if self.noise_var is not None:
            self.bnn.sigma_noise = math.sqrt(self.noise_var)

    def _posthoc_laplace(self, train_loader):
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.n_epochs * len(train_loader)
        )
        loss_func = (
            nn.MSELoss() if self.likelihood == "regression" else nn.CrossEntropyLoss()
        )

        for _ in range(self.n_epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.net(x)
                loss = loss_func(output, y)
                loss.backward()
                optimizer.step()
                scheduler.step()

        self.net.eval()
        self.bnn = Laplace(
            self.net,
            self.likelihood,
            subset_of_weights=self.subset_of_weights,
            hessian_structure=self.hess_factorization,
            backend=self.backend,
            enable_backprop=self.enable_backprop,  # Important!
        )
        self.bnn.fit(train_loader)

        if self.likelihood == "classification":
            self.bnn.optimize_prior_precision(n_steps=self.posthoc_marglik_iters)
        else:
            # For regression, tune prior precision and observation noise
            log_prior, log_sigma = (
                torch.ones(1, requires_grad=True),
                torch.ones(1, requires_grad=True),
            )
            hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
            for _ in range(self.posthoc_marglik_iters):
                hyper_optimizer.zero_grad()
                neg_marglik = -self.bnn.log_marginal_likelihood(
                    log_prior.exp(), log_sigma.exp()
                )
                neg_marglik.backward()
                hyper_optimizer.step()

    def _get_train_loader(self):
        return data_utils.DataLoader(
            data_utils.TensorDataset(self.train_X, self.train_Y),
            batch_size=self.batch_size,
            shuffle=True,
        )
