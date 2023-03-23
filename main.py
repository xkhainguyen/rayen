import sys
import os
import json
import time
import pickle
import pathlib
import numpy as np
import torch
import torch.nn as nn
import argparse
import itertools
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader

# import utils
from models import BarycentricModel, ProjectionModel, WrapperWalkerForImages
from dataset_util import MNIST
from osqp_projection import BatchProjector

from linear_constraint_layer import LinearConstraintLayer
from create_dataset import createProjectionDataset, getCorridorDatasetsAndLC
from examples_sets import getExample

import utils

class SplittedDatasetAndGenerator():
	def __init__(self, dataset, percent_train, percent_val, batch_size):
		assert percent_train<=1
		assert percent_val<=1
		assert (percent_train+percent_val)<=1

		train_size = int(percent_train * len(dataset))
		val_size = int(percent_val * len(dataset))
		test_size = len(dataset) - train_size - val_size

		self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

		# assert len(self.train_dataset)>0
		# assert len(self.val_dataset)>0
		# assert len(self.test_dataset)>0

		utils.printInBoldRed(f"Elements [train, val, test]={[len(self.train_dataset), len(self.val_dataset), len(self.test_dataset)]}")

		self.batch_size=batch_size

		if(len(self.train_dataset)>0):
			self.train_generator = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
			utils.printInBoldRed(f"Train batches {len(self.train_generator)}")


		if(len(self.val_dataset)>0):
			self.val_generator = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
			utils.printInBoldRed(f"Val batches {len(self.val_generator)}")


		if(len(self.test_dataset)>0):
			self.test_generator = DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False) #One batch only for testing [better inference time estimation]
			utils.printInBoldRed(f"Test batches {len(self.test_generator)}")


		# utils.printInBoldRed(f"Created DataLoader with batches [train, val, test]={[len(self.train_generator), len(self.val_generator), len(self.test_generator)]}")

	

def append_filename(params, filename):
	dir_path = os.path.join(params['result_dir'], params['method'])
	pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
	return os.path.join(dir_path, filename)

def train_model(model, params, sdag):
	device_id = params['device']
	device = torch.device('cuda:{}'.format(device_id) if device_id >= 0 else 'cpu')
	# print('Using device:\n', device)
	model = model.to(device)
	# print(model)

	optimizer = torch.optim.Adam(model.parameters(),lr=params['learning_rate'])


	# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, cooldown=5)

	loss_fn = nn.MSELoss(reduction='mean')

	metrics = {'train_loss_per_sample': [], 'val_loss_per_sample': []}
	
	#See https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
	for epoch in range(params['num_epochs']): #Loop over the dataset multiple times
		model.train()
		sum_all_squares = 0
		######################### TRAIN
		for x, y in sdag.train_generator:
			optimizer.zero_grad()

			#----------------------
			x = x.to(device)
			y = y.to(device)
			y_predicted = model(x)
			loss = loss_fn(y_predicted, y)
			sum_all_squares = sum_all_squares + loss.item()*torch.numel(x);
			#----------------------

			# assert abs(loss.item()*torch.numel(x)-nn.MSELoss(reduction='sum')(y_predicted, y))<1e-7

			loss.backward()
			optimizer.step()


		metrics['train_loss_per_sample'].append(sum_all_squares/len(sdag.train_generator.dataset))


		######################### VALIDATION
		with torch.set_grad_enabled(False):
			model.eval()
			sum_all_squares = 0
			for x, y in sdag.val_generator:

				#----------------------
				x = x.to(device)
				y = y.to(device)
				y_predicted = model(x)
				loss = loss_fn(y_predicted, y)
				sum_all_squares = sum_all_squares + loss.item()*torch.numel(x);
				#----------------------


			metrics['val_loss_per_sample'].append(sum_all_squares/len(sdag.val_generator.dataset))

		if epoch % params['verbosity'] == 0:
			print('{}: train: {}, val: {}, lr: {:.2E}'.format(
				epoch,
				metrics['train_loss_per_sample'][-1],
				metrics['val_loss_per_sample'][-1],
				optimizer.param_groups[0]['lr']))

	return metrics

def test_model(model, params, sdag, lc):
	device_id = params['device']
	device = torch.device('cuda:{}'.format(device_id) if device_id >= 0 else 'cpu')
	model = model.to(device)

	model.eval()
	val_loss = 0

	violations = np.empty((0,1))

	loss_fn = nn.MSELoss(reduction='mean')

	with torch.set_grad_enabled(False):
		sum_all_squares = 0.0
		total_time_s=0.0;
		for x, y in sdag.test_generator:

			#----------------------
			x = x.to(device)
			y = y.to(device)
			start_time = time.time()
			y_predicted = model(x)
			total_time_s = total_time_s + (time.time() -start_time)
			loss = loss_fn(y_predicted, y)
			sum_all_squares = sum_all_squares + loss.item()*torch.numel(x);
			#----------------------

			violations_of_this_batch=np.apply_along_axis(lc.getViolation,axis=1, arr=y_predicted.cpu().numpy())
			#Shape of violations_of_this_batch is [batch_size, 1]
			violations=np.concatenate((violations, violations_of_this_batch), axis=0)

	
	assert np.all(violations>=0) #violations, by definition are nonnegative
	assert violations.shape[0]==len(sdag.test_dataset), f"violations.shape[0]={violations.shape[0]}, len(sdag.test_dataset)={len(sdag.test_dataset)}"
	
	num_samples=len(sdag.test_dataset);
	metrics = {"test_loss_per_sample": sum_all_squares/num_samples,
		       "violation_per_sample": np.mean(violations),
		       "total_time_s_per_sample": total_time_s/num_samples}
	
	return metrics

def main(params):

	torch.set_default_dtype(torch.float64) ##Use float32 here??

	## PROJECTION EXAMPLES
	# lc=getExample(1)
	# my_dataset=createProjectionDataset(200, lc, 4.0);
	# my_dataset_out_dist=createProjectionDataset(200, lc, 7.0);

	## CORRIDOR EXAMPLES
	my_dataset, my_dataset_out_dist, lc=getCorridorDatasetsAndLC()

	sdag=SplittedDatasetAndGenerator(my_dataset, percent_train=0.6, percent_val=0.2, batch_size=params['batch_size'])
	sdag_out_dist=SplittedDatasetAndGenerator(my_dataset_out_dist, percent_train=0.0, percent_val=0.0, batch_size=params['batch_size'])

	results = []  # a list of dicts
	# for trial in range(params['n_trials']):

	######################### TRAINING
	model = LinearConstraintLayer(lc, method=params['method']) 
	mapper=utils.create_mlp(input_dim=my_dataset.getNumelX(), output_dim=model.getNumelOutputMapper(), net_arch=[64,64])
	# mapper=nn.Sequential() #do nothing.
	model.setMapper(mapper)


	training_metrics = train_model(model, params, sdag)
	print("Testing model...")
	testing_metrics = test_model(model, params, sdag, lc)
	testing_metrics_out_dist = test_model(model, params, sdag_out_dist, lc)
	print("Model tested...")

	#####PRINT STUFF
	print("\n\n-----------------------------------------------------")
	print(f"Method={params['method']}")
	print(f"Num of trainable params={sum(	p.numel() for p in model.parameters() if p.requires_grad)}")
	utils.printInBoldBlue(f"Training: \n"\
						  f"    loss: {training_metrics['train_loss_per_sample'][-1]} \n"\
						  f"    val loss: {training_metrics['val_loss_per_sample'][-1]}")

	######################### TESTING
	utils.printInBoldRed(f"Testing: \n"\
						 f"  loss: {testing_metrics['test_loss_per_sample']} \n"\
						 f"  violation: {testing_metrics['violation_per_sample']} \n"\
						 f"  total_time_ms_per_sample: {1000.0*testing_metrics['total_time_s_per_sample']}")

	utils.printInBoldRed(f"Testing outside distrib: \n"\
						 f"  loss: {testing_metrics_out_dist['test_loss_per_sample']} \n"\
						 f"  violation: {testing_metrics_out_dist['violation_per_sample']} \n"\
						 f"  total_time_ms_per_sample: {1000.0*testing_metrics_out_dist['total_time_s_per_sample']}")


if __name__ == '__main__':


	parser = argparse.ArgumentParser()
	parser.add_argument('--method', type=str, default='walker') #walker or barycentric or unconstrained
	parser.add_argument('--result_dir', type=str, default='results')
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--num_epochs', type=int, default=1000)
	# parser.add_argument('--n_trials', type=int, default=1)
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--n_evals', type=int, default=1)
	parser.add_argument('--verbosity', type=int, default=1)
	parser.add_argument('--learning_rate', type=float, default=1e-3)
	parser.add_argument('--log_to_file', dest='log_to_file', action='store_true', default=True)
	parser.add_argument('--box_constraints', dest='box_constraints', action='store_true', default=False)
	args = parser.parse_args()
	params = vars(args)

	print('Parameters:\n', params)

	main(params)