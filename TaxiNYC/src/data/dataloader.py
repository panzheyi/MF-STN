import os
import h5py
import logging
import mxnet as mx
import numpy as np
import pandas as pd
import math
import multiprocessing as mp

from data import utils
from config import DATA_PATH, TRAIN_PROP, EVAL_PROP
from config import ROWS, COLUMES 
from config import FLOW_INPUT_LEN, FLOW_OUTPUT_LEN

class RandomSampler(mx.gluon.data.Sampler):
	def __init__(self, length, batch_size, iterations_per_epoch):
		self._length = length
		self._batch_size = batch_size
		self._iterations_per_epoch = iterations_per_epoch
	
	def __iter__(self):
		for i in range(self._iterations_per_epoch):
			yield np.random.choice(self._length, self._batch_size)
	
	def __len__(self):
		return self._iterations_per_epoch * self._length
	
def load_flow():
	data = utils.load_h5(os.path.join(DATA_PATH, 'NYC_FLOW.h5'), ['data'])
	print('data shape', data.shape)

	days = data.shape[0]

	n_timestamp = data.shape[0]
	num_train = int(n_timestamp * TRAIN_PROP)
	num_eval = int(n_timestamp * EVAL_PROP)
	num_test = n_timestamp - num_train - num_eval

	return data[:num_train], data[num_train: num_train + num_eval], data[-num_test:]

def create_dataset_flow(flow, scaler, sampler_type, batch_size, iterations_per_epoch=None):
	n_timestamp_per_day = flow.shape[1]
	flow = np.reshape(flow, (-1, ROWS, COLUMES, flow.shape[-1]))

	mask = np.sum(flow, axis=(1,2,3)) > 5000

	flow = scaler.transform(flow)
	n_timestamp, rows, cols, _ = flow.shape

	timespan = (np.arange(n_timestamp) % n_timestamp_per_day) / float(n_timestamp_per_day)

	timespan = np.tile(timespan, (1, rows, cols, 1)).T
	flow = np.concatenate((flow, timespan), axis=3)

	data, label = [], []
	for i in range(n_timestamp - FLOW_INPUT_LEN - FLOW_OUTPUT_LEN + 1):
		if mask[i + FLOW_INPUT_LEN: i + FLOW_INPUT_LEN + FLOW_OUTPUT_LEN].sum() != FLOW_OUTPUT_LEN:
			continue

		data.append(flow[i: i + FLOW_INPUT_LEN])
		label.append(flow[i + FLOW_INPUT_LEN: i + FLOW_INPUT_LEN + FLOW_OUTPUT_LEN])

		if i % 1000 == 0:
			logging.info('[Flow] processed %d timestamps.', i)

	data = mx.nd.array(np.stack(data)) # [B, T, N, D]
	label = mx.nd.array(np.stack(label)) # [B, T, N, D]

	logging.info('shape of flow data: %s', data.shape)
	logging.info('shape of flow label: %s', label.shape)

	n = data.shape[0]
	sampler = None
	if sampler_type == 'random':
		sampler = RandomSampler(n, batch_size=batch_size, iterations_per_epoch=iterations_per_epoch)
	elif sampler_type == 'batch':
		sampler = mx.gluon.data.SequentialSampler(n)
		sampler = mx.gluon.data.BatchSampler(sampler, batch_size=batch_size, last_batch='rollover')
	else:
		raise Exception('unknown sampler type!')

	dataset = mx.gluon.data.ArrayDataset(data, label)
	return mx.gluon.data.DataLoader(dataset, batch_sampler=sampler, num_workers=4)

def dataloader_flow(settings):
	settings = settings['training']

	train, eval, test = load_flow()
	scaler = utils.Scaler()
	scaler.fit(train)

	flow_train = create_dataset_flow(train, scaler, sampler_type='random', batch_size=settings['flow_batch_size'], iterations_per_epoch=settings['iterations_per_epoch'])
	flow_eval = create_dataset_flow(eval, scaler, sampler_type='batch', batch_size=settings['flow_batch_size'])
	flow_test = create_dataset_flow(test, scaler, sampler_type='batch', batch_size=settings['flow_batch_size'])

	return flow_train, flow_eval, flow_test, scaler

def dataloader_flow_stdn(settings):
	from data import dataloader_stdn
	return dataloader_stdn.dataloader_flow_stdn(settings)