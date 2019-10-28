import os
import h5py
import numpy as np
import pandas as pd
import multiprocessing as mp

class Scaler:
	def __init__(self, mean=None, std=None):
		self.mean = mean
		self.std = std

	def fit(self, data):
		self.mean = np.mean(data)
		self.std = np.std(data)

	def set_mean(self, mean):
		self.mean = mean
	
	def set_std(self, std):
		self.std = std
	
	def transform(self, data):
		return (data - self.mean) / self.std
	
	def inverse_transform(self, data):
		return data * self.std + self.mean

def load_h5(filename, keywords):
	f = h5py.File(filename, 'r')
	data = []
	for name in keywords:
		data.append(np.array(f[name]))
	f.close()
	if len(data) == 1:
		return data[0]
	return data

def write_h5(filename, d):
	f = h5py.File(filename, 'w')
	for key, value in d.items():
		f.create_dataset(key, data=value)
	f.close()

def from_str_to_np(s):
	arr = np.fromstring(s, dtype=np.int32, sep=',')
	arr = arr.reshape(-1, 3)
	return arr

def load_trajectory(filename):
	with open(filename) as f:
		lines = f.readlines()
	
	pool = mp.Pool()
	trajectory = pool.map(from_str_to_np, lines)
	return trajectory


def fill_missing(data):
	T, N, D = data.shape
	data = np.reshape(data, (T, N * D))
	df = pd.DataFrame(data)
	df = df.fillna(method='pad')
	df = df.fillna(method='bfill')
	data = df.values
	data = np.reshape(data, (T, N, D))
	data[np.isnan(data)] = 0
	return data