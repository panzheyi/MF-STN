import numpy as np
import mxnet as mx
from mxnet import nd

from data.utils import Scaler
from config import ROWS, COLUMES

class Metric:
	def __init__(self, name):
		self.name = name
		self.cnt = None
		self.loss = None
	
	def reset(self):
		self.cnt = None
		self.loss = None
	
	def get_value(self):
		raise NotImplementedError()

class MAE(Metric):
	def __init__(self, scaler, pred_name, label_name, name='mae'):
		super(MAE, self).__init__(name)
		self.scaler = scaler
		self.pred_name = pred_name
		self.label_name = label_name
	
	def update(self, data):
		pred = self.scaler.inverse_transform(data[self.pred_name])
		label = self.scaler.inverse_transform(data[self.label_name])

		_cnt = pred.size
		_loss = nd.sum(nd.abs(pred - label)).as_in_context(mx.cpu())
		if self.cnt is None:
			self.cnt = self.loss = 0
		
		self.cnt += _cnt
		self.loss += _loss
	
	def get_value(self):
		return { self.name: self.loss / (self.cnt + 1e-8) }

class RMSE(Metric):
	def __init__(self, scaler, pred_name, label_name, name='rmse'):
		super(RMSE, self).__init__(name)
		self.scaler = scaler
		self.pred_name = pred_name
		self.label_name = label_name
	
	def update(self, data):
		pred = self.scaler.inverse_transform(data[self.pred_name])
		label = self.scaler.inverse_transform(data[self.label_name])

		_cnt = pred.size
		_loss = nd.sum((pred - label) ** 2).as_in_context(mx.cpu())
		if self.cnt is None:
			self.cnt = self.loss = 0
		
		self.cnt += _cnt
		self.loss += _loss

	def get_value(self):
		return { self.name: nd.sqrt(self.loss / (self.cnt + 1e-8)) }

class MAPE(Metric):
	def __init__(self, scaler, pred_name, label_name, name='smape'):
		super(MAPE, self).__init__(name)
		self.scaler = scaler
		self.pred_name = pred_name
		self.label_name = label_name

	def update(self, data):
		pred = self.scaler.inverse_transform(data[self.pred_name])
		label = self.scaler.inverse_transform(data[self.label_name])
		mask = label >= 10

		_cnt = nd.sum(mask).as_in_context(mx.cpu())
		_loss = nd.sum(nd.abs(pred - label) / (nd.abs(label) + 1e-5) * mask).as_in_context(mx.cpu())
		if self.cnt is None:
			self.cnt = self.loss = 0
		
		self.cnt += _cnt
		self.loss += _loss

	def get_value(self):
		return { self.name: self.loss / (self.cnt + 1e-8) }


class SMAPE(Metric):
	def __init__(self, scaler, pred_name, label_name, name='smape'):
		super(SMAPE, self).__init__(name)
		self.scaler = scaler
		self.pred_name = pred_name
		self.label_name = label_name

	def update(self, data):
		pred = self.scaler.inverse_transform(data[self.pred_name])
		label = self.scaler.inverse_transform(data[self.label_name])
		mask = label >= 10

		_cnt = nd.sum(mask).as_in_context(mx.cpu())
		_loss = nd.sum(nd.abs(pred - label) / (nd.abs(pred) + nd.abs(label) + 1e-5) * mask).as_in_context(mx.cpu())
		if self.cnt is None:
			self.cnt = self.loss = 0
		
		self.cnt += _cnt
		self.loss += _loss

	def get_value(self):
		return { self.name: self.loss * 2.0 / (self.cnt + 1e-8) }