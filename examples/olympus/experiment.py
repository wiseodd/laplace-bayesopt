import numpy as np
import torch
import os
from olympus import Olympus, Campaign, Dataset
from .laplace_olympus import LaplacePlanner
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--method', choices=['LA', 'GF', 'GP', 'RS'], default='LA')
parser.add_argument('--dataset', default='alkox')
parser.add_argument('--exp_len', type=int, default=500)
parser.add_argument('--cuda', default=False, action='store_true')
parser.add_argument('--randseed', type=int, default=9999)
args = parser.parse_args()

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

planners = {
    'LA': LaplacePlanner(goal=Dataset(kind=args.dataset).goal, num_init_design=20),
    'GF': 'Gryffin',
    'GP': 'Botorch',
    'RS': 'RandomSearch'
}

planner = planners[args.method]
campaign = Campaign()

start_time = time.time()
Olympus().run(
    planner=planner, dataset=args.dataset,
    num_iter=args.exp_len, campaign=campaign,
    database=None
)
end_time = time.time()

values = campaign.observations.get_values().flatten()
time_elapsed = end_time - start_time
results = {
    'vals': values,
    'time_elapsed': time_elapsed
}

path = f'results/{args.dataset}'
if not os.path.exists(path):
    os.makedirs(path)

np.save(f'{path}/{args.method}_{args.randseed}.npy', results)
