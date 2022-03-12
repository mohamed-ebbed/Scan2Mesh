import torch
import tqdm
import random
import numpy as np
import pickle

from training import edge_train

import argparse
import json


parser = argparse.ArgumentParser(description='Arguments for training scan2mesh')

parser.add_argument('--config', type=str, help='The experiment config', required=True)

args = parser.parse_args()



with open(args.config, 'r') as f:
  config = json.load(f)

edge_train.main(config)