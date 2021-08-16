#!/usr/bin/env python

from options.train_options import TrainOptions
import sys

class Options:
	def __init__(self, opt_dict):
		for key, val in opt_dict.items():
			setattr(self, key, val)


if __name__ == '__main__':
	opt = TrainOptions().parse()
	opt_dict = vars(opt)
	
	for key, val in opt_dict.items():
		print(f'self.{key} = {val}')

	options = Options(opt_dict)

	# for attribute in dir(options):
	# 	print(attribute)
	
