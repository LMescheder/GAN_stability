import argparse
import os
from os import path
import copy
import numpy as np
import torch
from torch import nn
from gan_training import utils
from gan_training.checkpoints import CheckpointIO
from gan_training.distributions import get_ydist, get_zdist, interpolate_sphere
from gan_training.config import (
    load_config, build_models
)

# Arguments
parser = argparse.ArgumentParser(
    description='Create interpolations for a trained GAN.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()

config = load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)

# Shorthands
nlabels = config['data']['nlabels']
out_dir = config['training']['out_dir']
batch_size = config['test']['batch_size']
sample_size = config['test']['sample_size']
sample_nrow = config['test']['sample_nrow']
checkpoint_dir = path.join(out_dir, 'chkpts')
interp_dir = path.join(out_dir, 'test', 'interp_class')

# Creat missing directories
if not path.exists(interp_dir):
    os.makedirs(interp_dir)

# Logger
checkpoint_io = CheckpointIO(
    checkpoint_dir=checkpoint_dir
)

# Get model file
model_file = config['test']['model_file']

# Models
device = torch.device("cuda:0" if is_cuda else "cpu")

generator, discriminator = build_models(config)
print(generator)
print(discriminator)

# Put models on gpu if needed
generator = generator.to(device)
discriminator = discriminator.to(device)

# Use multiple GPUs if possible
generator = nn.DataParallel(generator)
discriminator = nn.DataParallel(discriminator)

# Register modules to checkpoint
checkpoint_io.register_modules(
    generator=generator,
    discriminator=discriminator,
)

# Test generator
if config['test']['use_model_average']:
    generator_test = copy.deepcopy(generator)
    checkpoint_io.register_modules(generator_test=generator_test)
else:
    generator_test = generator

# Distributions
ydist = get_ydist(nlabels, device=device)
zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
                  device=device)


# Load checkpoint if existant
load_dict = checkpoint_io.load(model_file)
it = load_dict.get('it', -1)
epoch_idx = load_dict.get('epoch_idx', -1)

# Interpolations
print('Creating interplations...')
nsubsteps = config['interpolations']['nsubsteps']
ys = config['interpolations']['ys']

nsteps = len(ys)
z = zdist.sample((sample_size,))
ts = np.linspace(0, 1, nsubsteps)

it = 0
for y1, y2 in zip(ys, ys[1:] + [ys[0]]):
    for t in ts:
        y1_pt = torch.full((sample_size,), y1, dtype=torch.int64, device=device)
        y2_pt = torch.full((sample_size,), y2, dtype=torch.int64, device=device)
        y1_embed = generator_test.module.embedding(y1_pt)
        y2_embed = generator_test.module.embedding(y2_pt)
        t = float(t)
        y_embed = (1 - t) * y1_embed + t * y2_embed
        with torch.no_grad():
            x = generator_test(z, y_embed)
            utils.save_images(x, path.join(interp_dir, '%04d.png' % it),
                              nrow=sample_nrow)
            it += 1
            print('%d/%d done!' % (it, nsteps * nsubsteps))
