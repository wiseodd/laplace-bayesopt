from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
import pytest
import torch
from torch import nn
from laplace_bayesopt.botorch import LaplaceBoTorch


torch.manual_seed(9999)
torch.set_default_tensor_type(torch.DoubleTensor)


@pytest.fixture
def net_singletask():
    net = torch.nn.Sequential(
        nn.Linear(3, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU(), nn.Linear(20, 1)
    )
    return net


@pytest.fixture
def net_multitask():
    net = torch.nn.Sequential(
        nn.Linear(3, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU(), nn.Linear(20, 5)
    )
    return net


@pytest.fixture
def reg_data_singletask():
    X = torch.randn(10, 3)
    y = torch.randn(10, 1)
    return X, y


@pytest.fixture
def reg_data_multitask():
    X = torch.randn(10, 3)
    y = torch.randn(10, 5)
    return X, y


def test_posterior_singletask(net_singletask, reg_data_singletask):
    get_nn = lambda: net_singletask
    train_X, train_Y = reg_data_singletask
    model = LaplaceBoTorch(get_nn, train_X, train_Y)

    # Batch, joint-dim, feature-dim, num_task
    B, Q, D, K = 15, 4, train_X.shape[-1], 1
    post = model.posterior(torch.randn(B, Q, D))

    assert isinstance(post.mvn, MultivariateNormal)
    assert post.mean.shape == (B, Q * K, 1)  # Quirk of GPyTorch
    assert post.covariance_matrix.shape == (B, Q * K, Q * K)


def test_posterior_multitask(net_multitask, reg_data_multitask):
    get_nn = lambda: net_multitask
    train_X, train_Y = reg_data_multitask
    model = LaplaceBoTorch(get_nn, train_X, train_Y)

    # Batch, joint-dim, feature-dim, num_task
    B, Q, D, K = 15, 4, train_X.shape[-1], train_Y.shape[-1]
    post = model.posterior(torch.randn(B, Q, D))

    assert isinstance(post.mvn, MultitaskMultivariateNormal)
    assert post.mean.shape == (B, Q, K)
    assert post.covariance_matrix.shape == (B, Q * K, Q * K)


def test_condition_on_observations(net_multitask, reg_data_multitask):
    get_nn = lambda: net_multitask
    train_X, train_Y = reg_data_multitask
    model = LaplaceBoTorch(get_nn, train_X, train_Y)

    # Batch, joint-dim, feature-dim, num_task
    B, Q, D, K = 15, 4, train_X.shape[-1], train_Y.shape[-1]
    model_new = model.condition_on_observations(torch.randn(B, D), torch.randn(B, K))

    assert model_new.train_X.shape == (train_X.shape[0] + B, D)
    assert model_new.train_Y.shape == (train_Y.shape[0] + B, K)


def test_get_prediction(net_multitask, reg_data_multitask):
    get_nn = lambda: net_multitask
    train_X, train_Y = reg_data_multitask
    model = LaplaceBoTorch(get_nn, train_X, train_Y)

    # Batch, joint-dim, feature-dim, num_task
    B, Q, D, K = 15, 4, train_X.shape[-1], train_Y.shape[-1]

    Y_mean, Y_var = model._get_prediction(torch.randn(B, D), joint=False)
    assert Y_mean.shape == (B, K)
    assert Y_var.shape == (B, K, K)

    Y_mean, Y_var = model._get_prediction(
        torch.randn(B, D), joint=False, use_test_loader=True
    )
    assert Y_mean.shape == (B, K)
    assert Y_var.shape == (B, K, K)

    Y_mean, Y_var = model._get_prediction(torch.randn(B, D), joint=True)
    assert Y_mean.shape == (B * K,)
    assert Y_var.shape == (B * K, B * K)

    Y_mean, Y_var = model._get_prediction(
        torch.randn(B, D), joint=True, use_test_loader=True
    )
    assert Y_mean.shape == (B * K,)
    assert Y_var.shape == (B * K, B * K)
