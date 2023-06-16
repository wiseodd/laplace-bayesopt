import numpy as np
import torch
from torch import nn
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.optim import (
    optimize_acqf,
    optimize_acqf_discrete,
    optimize_acqf_mixed,
)
from olympus.objects import ParameterVector
from olympus.planners import AbstractPlanner
from olympus import Logger

from olympus.planners import Botorch
from olympus.planners.planner_botorch.botorch_utils import *

from surrogate.laplace import LaplaceBNN


class LaplacePlanner(Botorch):

    def _ask(self):
        if len(self._values) < self.num_init_design:
            # sample using initial design strategy
            sample, raw_sample = propose_randomly(1, self.param_space, self.use_descriptors)
            return_params = ParameterVector().from_array(
                raw_sample[0], self.param_space
            )

        else:
            self.train_x_scaled, self.train_y_scaled = self.build_train_data()

            # Assume all x's are fully continuous for now
            # The BNN is fitted upon construction
            def get_nn():
                return torch.nn.Sequential(
                    nn.Linear(self.train_x_scaled.shape[-1], 50),
                    nn.GELU(),
                    nn.Linear(50, 50),
                    nn.GELU(),
                    nn.Linear(50, self.train_x_scaled.shape[-1])
                )

            model = LaplaceBNN(
                get_nn=get_nn,
                train_X=self.train_x_scaled,
                train_Y=self.train_y_scaled,
            )

            # get the incumbent point
            f_best_argmin = torch.argmin(self.train_y_scaled)
            f_best_scaled = self.train_y_scaled[f_best_argmin][0].float()
            f_best_raw = self._values[f_best_argmin][0]

            acqf = ExpectedImprovement(
                model, f_best_scaled, objective=None, maximize=False
            )  # always minimization in Olympus

            bounds = get_bounds(self.param_space, self.has_descriptors)
            choices_feat, choices_cat = None, None

            # Assume fully continuous for now
            results, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                num_restarts=200,
                q=self.batch_size,
                raw_samples=1000,
            )

            # convert the results form torch tensor to numpy
            results_np = np.squeeze(results.detach().numpy())

            if (
                not self.problem_type == 'fully_categorical'
                and not self.has_descriptors
            ):
                # reverse transform the inputs
                results_np = reverse_normalize(
                    results_np, self._mins_x, self._maxs_x
                )

            if choices_feat is not None:
                choices_feat = reverse_normalize(
                    choices_feat, self._mins_x, self._maxs_x
                )

            # project the sample back to Olympus format
            sample = project_to_olymp(
                results_np,
                self.param_space,
                has_descriptors=self.has_descriptors,
                choices_feat=choices_feat,
                choices_cat=choices_cat,
            )
            return_params = ParameterVector().from_dict(
                sample, self.param_space
            )

        return return_params
