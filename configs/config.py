import yaml

class options:
	def __init__(self, option=None, verbose=False):
   
		# basic attribute
		self.gpu_ids = [0]
		self.checkpoints_dir = './checkpoints'
		self.model = 'cycle_gan'
		self.input_nc = 1
		self.output_nc = 1
		self.ngf = 64
		self.ndf = 64
		self.netD = 'basic'
		self.netG = 'resnet_9blocks'
		self.n_layers_D = 3
		self.norm = 'instance'
		self.init_type = 'normal'
		self.init_gain = 0.02
		self.no_dropout = True
		self.dataset_mode = 'toyzero'
		self.direction = 'AtoB'
		self.serial_batches = False
		self.num_threads = 1
		self.batch_size = 32
		self.load_size = 286
		self.crop_size = 256
		self.max_dataset_size = 1000
		self.preprocess = 'resize_and_crop'
		self.no_flip = False
		self.display_winsize = 256
		self.epoch = 'latest'
		self.load_iter = 0
		self.verbose = False
		self.suffix = ''
		self.display_freq = 400
		self.display_ncols = 4
		self.display_id = 1
		self.display_server = 'http://localhost'
		self.display_env = 'main'
		self.display_port = 8097
		self.update_html_freq = 1000
		self.print_freq = 100
		self.no_html = False
		self.save_latest_freq = 5000
		self.save_epoch_freq = 5
		self.save_by_iter = False
		self.continue_train = False
		self.epoch_count = 1
		self.phase = 'train'
		self.n_epochs = 200
		self.n_epochs_decay = 100
		self.beta1 = 0.5
		self.lr = 0.0002
		self.gan_mode = 'lsgan'
		self.pool_size = 50
		self.lr_policy = 'linear'
		self.lr_decay_iters = 50
		self.lambda_A = 10.0
		self.lambda_B = 10.0
		self.lambda_identity = 0.5
		self.isTrain = True
   
   
		if option is not None:
			if isinstance(option, dict):
				self.update(option_dict, verbose=verbose)
			elif isinstance(option, str):
				self.update_from_yaml(option, verbose=verbose)
				
		assert hasattr(self, 'dataroot'), "options must have attribute dataroot"
		assert hasattr(self, 'name'), "options must have attribute name for saving checkpoints"
		assert (self.phase == 'train' and self.isTrain) or (self.phase == 'test' and not self.isTrain)

	def update(self, option_dict, verbose=True):
		for key, val in option_dict.items():
			if hasattr(self, key):
				old_val = getattr(self, key)
				if old_val != val:
					setattr(self, key, val)
					if verbose:
						print(f'updated attribute:\n\t{key}: {old_val} -> {val}')
			else:
				setattr(self, key, val)
				if verbose:
					print(f'new attribute:\n\t{key} = {val}')
   
	def to_yaml(self, yaml_fname):
		with open(yaml_fname, 'w') as handle:
			yaml.dump(vars(self), handle)
   
	def update_from_yaml(self, yaml_fname, verbose=False):
		with open(yaml_fname, 'r') as handle:
			option_dict = yaml.safe_load(handle)
			self.update(option_dict, verbose=verbose)
   
	def print_options(self):
		for key, val in vars(self).items():
			print(f'self.{key}={val}')

if __name__ == "__main__":
	opt = options('config_toyzero_512x512_2_64.yaml', verbose=True)
	opt.print_options()
