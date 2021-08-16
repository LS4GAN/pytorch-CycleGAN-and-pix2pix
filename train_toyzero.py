#!/usr/bin/env python

# Training CycleGAN with Toyzero data
import sys
import time
from pathlib import Path
import numpy as np
from collections import OrderedDict
import torch

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from configs.config import options 


# Load options
config_file = sys.argv[1]
opt = options(config_file)
opt.print_options()

# ans = input('Everything looks good? (Y/n)')
# if ans != 'Y':
# 	exit()
# if ans != 'Y':
# 	while True:
# 		parameter = input('parameter to update:\t')
# 		if hasattr(opt, parameter):
# 			old_val = getattr(opt, parameter)
# 			ptype = type(old_val)
# 			value = input('updated value:\t')
# 			if ptype == int:
# 				value = int(value)
# 			elif ptype == float:
# 				value = float(value)
# 			setattr(opt, parameter, value)
# 			print(f'\t{parameter}: {old_val} -> {getattr(opt, parameter)}')
# 			
# 		else:
# 			print(f'parameter {parameter} is not an attribute of options')
# 		ans = input('Done with updating? (Y/n)')
# 		if ans == 'Y':
# 			break
		
# make pt folder
pt_folder = Path(opt.checkpoints_dir)/opt.name
print(f'pt_folder = {pt_folder}')
if not pt_folder.exists():
	pt_folder.mkdir(parents=True)


# Load training and validation dataset
## Load training dataset
dataset = create_dataset(opt)
print(f'The number of training images = {len(dataset)}')
## Load aligned validation dataset
opt_valid = options(config_file)
opt_valid.dataset_mode = 'toyzero_aligned'
opt_valid.phase = 'test'
## let batch_size = max_dataset_size so that we don't have 
## to iterate through the loader.
opt_valid.batch_size = 100 
opt_valid.max_dataset_size = 100
dataset_valid = create_dataset(opt_valid)
print(f'The number of validation images = {len(dataset_valid)}')
dataset_valid = next(iter(dataset_valid))


# Load model
model = create_model(opt)	  # create a model given opt.model and other options
model.setup(opt)   # regular setup: load and print networks; create schedulers


# Train
def get_validation_visuals(model, dataset_valid):
	with torch.no_grad():
		real_A = dataset_valid['A'].to(model.device)
		fake_B = model.netG_A(real_A)
		rec_A = model.netG_B(fake_B)
		
		real_B = dataset_valid['B'].to(model.device)
		fake_A = model.netG_B(real_B)
		rec_B = model.netG_A(fake_A)
	
		idt_A = model.netG_A(real_B)
		idt_B = model.netG_B(real_A)
		
	visual_ret = OrderedDict()
	visual_ret = {
		'real_A': real_A, 'fake_B': fake_B, 'rec_A': rec_A,
		'real_B': real_B, 'fake_A': fake_A, 'rec_B': rec_B,
		'idt_A': idt_A, 'idt_B': idt_B
	}
	return visual_ret


def get_validation_losses(model, dataset_valid):
	visuals = get_validation_visuals(model, dataset_valid)
	
	criterion = torch.nn.L1Loss()
	with torch.no_grad():
		loss_trans_A = criterion(visuals['fake_A'], visuals['real_A'])
		loss_trans_B = criterion(visuals['fake_B'], visuals['real_B'])
		loss_cycle_A = criterion(visuals['rec_A'], visuals['real_A'])
		loss_cycle_B = criterion(visuals['rec_B'], visuals['real_B'])
		loss_idt_A = criterion(visuals['idt_B'], visuals['real_A'])
		loss_idt_B = criterion(visuals['idt_A'], visuals['real_B'])
		
		diff = criterion(visuals['real_A'], visuals['real_B'])
	
	loss_ret = {
		'loss_trans_A': loss_trans_A, 'loss_trans_B': loss_trans_B,
		'loss_cycle_A': loss_cycle_A, 'loss_cycle_B': loss_cycle_B,
		'loss_idt_A': loss_idt_A, 'loss_idt_B': loss_idt_B,
		'diff': diff
	}
	return loss_ret


class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKCYAN = '\033[96m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'


for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
	
	print(f'{bcolors.BOLD}\nStart of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay}{bcolors.ENDC}')
	
	# Train
	epoch_start_time = time.time()  # Epoch timer	
	for i, data in enumerate(dataset):  # Iter over minibatches
		model.set_input(data)		 
		model.optimize_parameters()   # Feed forward and backpropagation
	
	# Print loss and save model
	losses = model.get_current_losses()
	print(f"\tDiscriminator loss:\tD_A={losses['D_A']:.6f}, \tD_B={losses['D_B']:.6f}")
	print(f"\tGenerator loss:\t\tG_A={losses['G_A']:.6f}, \tG_A={losses['G_B']:.6f}")
	print(f"\tCycle loss:\t\tC_A={losses['cycle_A']:.6f}, \tC_B={losses['cycle_B']:.6f}")
	print(f"\tIdentity loss:\t\tI_A={losses['idt_A']:.6f}, \tI_B={losses['idt_B']:.6f}")
	print(f'\tTime Taken:\t\t{time.time() - epoch_start_time:.0f} sec')
	
	losses_valid = get_validation_losses(model, dataset_valid)
	print(f"{bcolors.OKCYAN}\n\tValidation losses:{bcolors.ENDC}")
	print(f"{bcolors.OKCYAN}\tDifference:\t{losses_valid['diff']:.6f}{bcolors.ENDC}")
	print(f"{bcolors.OKCYAN}\tTranslation:\tA={losses_valid['loss_trans_A']:.6f}, \tB={losses_valid['loss_trans_B']:.6f}{bcolors.ENDC}")
	print(f"{bcolors.OKCYAN}\tCycle:\t\tA={losses_valid['loss_cycle_A']:.6f}, \tB={losses_valid['loss_cycle_B']:.6f}{bcolors.ENDC}")
	print(f"{bcolors.OKCYAN}\tIdentity:\tA={losses_valid['loss_idt_A']:.6f}, \tB={losses_valid['loss_idt_B']:.6f}\n{bcolors.ENDC}")
	
	# Update learning rate
	model.update_learning_rate()
	
	# Check momory usage
	memory_cuda = torch.cuda.max_memory_allocated(device='cuda')
	print(f'\tpeak memory use:\t{memory_cuda/1024 ** 3:.3f}G')
	
	if epoch % opt.save_epoch_freq == 0:			  # cache our model every <save_epoch_freq> epochs
		print(f'{bcolors.OKGREEN}\tSaving models at the end of epoch {epoch}{bcolors.ENDC}')
		model.save_networks('latest')
		model.save_networks(epoch)


# def plot_results(visuals, num_samples=5):
# 	"""
# 	
# 	"""
# 	side = 3
# 	cmap = 'bwr'
# 	tensors = []
# 	keys = ['real_A', 'fake_A', 'idt_B', 'real_B', 'fake_B', 'idt_A']
# 	key_map = {
# 		'real_A': 'Real image from field A',
# 		'fake_A': 'Image translated from B to A ($G_{B \\rightarrow A}(b), b \in B$)',
# 		'idt_B': 'Image translated from A to A ($G_{B \\rightarrow A}(a), a \in A$)',
# 		'real_B': 'Real image from field B',
# 		'fake_B': 'Image translated from A to B ($G_{A \\rightarrow B}(a), a \in A$)',
# 		'idt_A': 'Image translated from B to B ($G_{A \\rightarrow B}(b), b \in B$)',
# 	}
# 	indices = None
# 	for key in keys:
# 		val = visuals[key]
# 		# print(key)
# 		if indices is None:
# 			indices = np.arange(len(val))
# 			indices = np.random.choice(indices, num_samples, replace=False)
# 		val = val[indices].cpu().detach().squeeze().numpy()
# 		
# 		fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * side * 1.1, side))
# 		for i, (ax, image) in enumerate(zip(axes, val)):
# 			vmin, vmax = image.min(), image.max()
# 			if vmin == 0:
# 				vmin = -.05
# 			if vmax == 0:
# 				vmax = .05
# 			divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
# 			ax.set_aspect(1)
# 			im = ax.pcolormesh(image, cmap=cmap, norm=divnorm)
# 			divider = make_axes_locatable(ax)
# 			cax = divider.append_axes('right', size='5%', pad=0.1)
# 			fig.colorbar(im, cax=cax, orientation='vertical')
# 		
# 		fig.suptitle(f"{key_map[key]}", fontsize=20)
# 		plt.tight_layout()
# 
# # plot_results(model.get_current_visuals(), num_samples=5)
# visuals = get_validation_visuals(model, dataset_valid)
# plot_results(visuals, num_samples=5)
