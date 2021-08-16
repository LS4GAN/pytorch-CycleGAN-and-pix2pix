import numpy as np
from pathlib import Path
# from torch.utils.data import Dataset, DataLoader
from data.base_dataset import BaseDataset



def load_image_fnames(dirname, max_dataset_size=float('inf')):
	"""
	load image fnames.
	If max_dataset_size is not infinity and is less than all available images,
	return a random subset of max_dataset_size image fnames.
	"""
	assert Path(dirname).exists(), f"{dirname} doesn't exist"
	image_fnames = np.array(sorted(list(Path(dirname).glob('*npz'))))
	
	if max_dataset_size != float('inf') and max_dataset_size < len(image_fnames):
		indices = np.arange(len(image_fnames))
		np.random.shuffle(indices)
		indices = indices[:max_dataset_size]
		image_fnames = image_fnames[indices]
	return image_fnames


class ToyzeroDataset(BaseDataset):
	def __init__(self, opt):
		BaseDataset.__init__(self, opt)
		dir_A = Path(opt.dataroot)/f'{opt.phase}A'
		dir_B = Path(opt.dataroot)/f'{opt.phase}B'
		self.image_fnames_A = load_image_fnames(dir_A, opt.max_dataset_size)
		self.image_fnames_B = load_image_fnames(dir_B, opt.max_dataset_size)
		self.size_A = len(self.image_fnames_A)
		self.size_B = len(self.image_fnames_B)
	
	def __len__(self):
		return max(self.size_A, self.size_B)
	
	def __load(self, image_fname):
		image = np.load(image_fname)
		image = image[image.files[0]]
		image = np.expand_dims(np.float32(image), 0)
		return image
	
	def __getitem__(self, index):
		index_A = index % self.size_A
		if self.opt.serial_batches:
			index_B = index % self.size_B
		else:
			index_B = np.random.randint(0, self.size_B - 1) # inclusive end
		path_A = self.image_fnames_A[index_A]
		path_B = self.image_fnames_B[index_B]
		image_A = self.__load(path_A)
		image_B = self.__load(path_B)
		
		return {'A': image_A, 'B': image_B, 'A_paths': str(path_A), 'B_paths': str(path_B)}

# class toyzero_dataset_options:
# 	"""
# 	I just list all the options here 
# 	"""
# 	def __init__(self):
# 		# data
# 		self.dataroot = '/sdcc/u/yhuang2/PROJs/GAN/datasets/ls4gan/toyzero_cropped/toyzero_2021-06-29_safi_U/'
# 		self.phase = 'train'
# 		assert self.phase in ['train', 'test'], "Invalid phase, choose from ['train', 'test']"
# 		self.max_dataset_size = 1000
# 		self.batch_size = 32
# 		self.num_workers = 1
# 
# 
# if __name__ == '__main__':
# 		
# 	opt = toyzero_dataset_options()
# 
# 	dataset = toyzero_dataset(opt)
# 	dataloader = DataLoader(
# 		dataset,
# 		batch_size=opt.batch_size,
# 		num_workers=opt.num_workers,
# 		shuffle=True
# 	)
# 	print(f'number of batches = {len(dataloader)}')
