import numpy as np
from pathlib import Path
# from torch.utils.data import Dataset, DataLoader
from data.base_dataset import BaseDataset



def load_image_fnames(dirname):
	"""
	load image fnames.
	If max_dataset_size is not infinity and is less than all available images,
	return a random subset of max_dataset_size image fnames.
	"""
	assert Path(dirname).exists(), f"{dirname} doesn't exist"
	image_fnames = np.array(sorted(list(Path(dirname).glob('*npz'))))
	return image_fnames


class ToyzeroAlignedDataset(BaseDataset):
	def __init__(self, opt):
		BaseDataset.__init__(self, opt)
		dir_A = Path(opt.dataroot)/f'{opt.phase}A'
		dir_B = Path(opt.dataroot)/f'{opt.phase}B'
		self.image_fnames_A = load_image_fnames(dir_A)
		self.image_fnames_B = load_image_fnames(dir_B)
		
		assert len(self.image_fnames_A) == len(self.image_fnames_B), "The dataset is not aligned"

		aligned = True
		for fname_A, fname_B in zip(self.image_fnames_A, self.image_fnames_B):
			if fname_A.stem != fname_B.stem:
				aligned = False
				break
		if not aligned:
			print("The dataset is not aligned")
			exit()
	
		self.size = len(self.image_fnames_A)
		
		if opt.max_dataset_size != float('inf') and opt.max_dataset_size < self.size:
			indices = np.arange(self.size)
			np.random.shuffle(indices)
			indices = indices[:opt.max_dataset_size]
			self.image_fnames_A = self.image_fnames_A[indices]
			self.image_fnames_B = self.image_fnames_B[indices]
			self.size = opt.max_dataset_size
	
	def __len__(self):
		return self.size
	
	def __load(self, image_fname):
		image = np.load(image_fname)
		image = image[image.files[0]]
		image = np.expand_dims(np.float32(image), 0)
		return image
	
	def __getitem__(self, index):
		if self.opt.serial_batches:
			index = index % self.size
		else:
			index = np.random.randint(0, self.size - 1)
		path_A = self.image_fnames_A[index]
		path_B = self.image_fnames_B[index]
		image_A = self.__load(path_A)
		image_B = self.__load(path_B)
		
		return {'A': image_A, 'B': image_B, 'A_paths': str(path_A), 'B_paths': str(path_B)}
