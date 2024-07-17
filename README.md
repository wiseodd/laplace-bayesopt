# Bayesian Optimization Interface for `laplace-torch`

## Installation

Install PyTorch first, then:

```
pip install --upgrade pip wheel packaging
pip install git+https://github.com/aleximmer/laplace.git@0.2
pip install laplace-bayesopt
```

## Usage

Basic usage

```python
from laplace_bayesopt.botorch import LaplaceBoTorch

def get_net():
    # Return a *freshly-initialized* PyTorch model
    return torch.nn.Sequential(
        ...
    )

# Initial X, Y pairs, e.g. obtained via random search
train_X, train_Y = ..., ...

model = LaplaceBoTorch(get_net, train_X, train_Y)

# Use this model in your existing BoTorch loop, e.g. to replace BoTorch's MultiTaskGP model.
```

The full arguments of `LaplaceBoTorch` can be found in the class documentation.

Check out a full BoTorch example in `examples/botorch/experiments.py`.

## Useful References

- General Laplace approximation: <https://arxiv.org/abs/2106.14806>
- Laplace for Bayesian optimization: <https://arxiv.org/abs/2304.08309>
- Benchmark of neural-net-based Bayesian optimizers: <https://arxiv.org/abs/2305.20028>
- The case for neural networks for Bayesian optimization: <https://arxiv.org/abs/2104.11667>

## Citation

```
@inproceedings{kristiadi2023promises,
  title={Promises and Pitfalls of the Linearized {L}aplace in {B}ayesian Optimization},
  author={Kristiadi, Agustinus and Immer, Alexander and Eschenhagen, Runa and Fortuin, Vincent},
  booktitle={AABI},
  year={2023}
}
```
