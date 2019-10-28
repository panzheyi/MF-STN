import os
import yaml
import random
import logging
logging.basicConfig(level=logging.INFO)

import mxnet as mx
from mxnet import nd, gluon, autograd, init

import numpy as np
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

import data.dataloader
from data.utils import Scaler
from config import PARAM_PATH
from helper.callback import Speedometer, Logger
from helper.metric import MAE, RMSE, SMAPE, MAPE
import model

def merge_metrics(m1, m2):
	if m1 is None: return m2
	if m2 is None: return m1
	return m1 + m2

def reset_metrics(metrics):
	if metrics is not None:
		for metric in metrics:
			metric.reset()

def update_metrics(metrics, results):
	if metrics is not None:
		for metric in metrics:
			for out in results:
				metric.update(out)

class ModelTrainer:
	def __init__(self, net, trainer, clip_gradient, logger, ctx):
		self.net = net
		self.trainer = trainer
		self.clip_gradient = clip_gradient
		self.logger = logger
		self.ctx = ctx
		
	def step(self, batch_size): # train step with gradient clipping
		self.trainer.allreduce_grads()
		grads = []
		for param in self.trainer._params:
			if param.grad_req == 'write':
				grads += param.list_grad()
		
		import math
		gluon.utils.clip_global_norm(grads, self.clip_gradient * math.sqrt(len(self.ctx)))
		self.trainer.update(batch_size, ignore_stale_grad=True)

	def train_model(self, epoch, dataloader, metrics):
		speedometer = Speedometer('[TRAIN]', epoch, frequent=50)
		speedometer.reset()

		reset_metrics(metrics)

		for nbatch, batch_data in enumerate(dataloader):
			inputs = [gluon.utils.split_and_load(x, self.ctx) for x in batch_data]

			with autograd.record():
				outputs = [self.net(*x) for x in zip(*inputs)]
			for out in outputs:
				out[0].backward()
			self.trainer.step(batch_data[0].shape[0])

			results = []
			for out in outputs:
				results += [out[1]]
			
			update_metrics(metrics, results)
			speedometer.log_metrics(nbatch + 1, metrics)

		speedometer.finish(metrics)

	def eval_by_func(self, func, epoch, dataloader, metrics, title):
		speedometer = Speedometer(title, epoch, frequent=50)
		speedometer.reset()

		reset_metrics(metrics)

		for nbatch, batch_data in enumerate(dataloader):
			inputs = [gluon.utils.split_and_load(x, self.ctx) for x in batch_data]
			outputs = [func(*x) for x in zip(*inputs)]

			results = []
			for out in outputs:
				results += [out[1]]
			
			update_metrics(metrics, results)
			speedometer.log_metrics(nbatch + 1, metrics)

		speedometer.finish(metrics)

	def fit(self, begin_epoch, num_epochs, train, eval, test, metrics=None):
		for epoch in range(begin_epoch, begin_epoch + num_epochs):
			if train is not None:
				self.train_model(epoch, train, metrics)

			if eval is not None:
				self.eval_by_func(self.net.forward, epoch, eval, metrics, '[EVAL]')
	
			if train is not None and len(metrics) > 0:
				self.logger.log(epoch, metrics)

			if test is not None:
				self.eval_by_func(self.net.forward, epoch, test, metrics, '[TEST]')


def main(args):
	with open(args.file, 'r') as f:
		settings = yaml.load(f)
	assert args.file[:-5].endswith(settings['model']['name']), \
		'The model name is not consistent! %s != %s' % (args.file[:-5], settings['model']['name'])

	mx.random.seed(settings['seed'])
	np.random.seed(settings['seed'])
	random.seed(settings['seed'])
	
	setting_dataset = settings['dataset']
	setting_model = settings['model']
	setting_train = settings['training']

	name = os.path.join(PARAM_PATH, setting_model['name'])
	model_type = getattr(model, setting_model['type'])
	net = model_type.net(settings)

	try:
		logger = Logger.load('%s.yaml' % name)
		net.load_parameters('%s-%04d.params' % (name, logger.best_epoch()), ctx=args.gpus)
		logger.set_net(net)
		print('Successfully loading the model %s [epoch: %d]' % (setting_model['name'], logger.best_epoch()))
	except:
		logger = Logger(name, net, setting_train['early_stop_metric'], setting_train['early_stop_epoch'])
		net.initialize(init.Xavier(), ctx=args.gpus)
		print('Initialize the model')

	num_params = 0
	for v in net.collect_params().values():
		num_params += np.prod(v.shape)
	print(net.collect_params())
	print('NUMBER OF PARAMS:', num_params)

	flow_train, flow_eval, flow_test, flow_scaler = getattr(data.dataloader, setting_dataset['flow'])(settings)
	
	model_trainer = ModelTrainer(
		net = net,
		trainer = gluon.Trainer(
			net.collect_params(),
			mx.optimizer.Adam(
				learning_rate	= setting_train['lr'],
				lr_scheduler	= mx.lr_scheduler.FactorScheduler(
					step			= setting_train['lr_decay_step'] * len(args.gpus),
					factor			= setting_train['lr_decay_factor'],
					stop_factor_lr	= 1e-6
				)
			),
			update_on_kvstore = False
		),
		clip_gradient = setting_train['clip_gradient'],
		logger = logger,
		ctx = args.gpus
	)

	flow_metrics = [
		MAE(scaler=flow_scaler, pred_name='flow_pred', label_name='flow_label', name='flow_mae'),
		RMSE(scaler=flow_scaler, pred_name='flow_pred', label_name='flow_label', name='flow_rmse'),
		MAPE(scaler=flow_scaler, pred_name='flow_pred', label_name='flow_label', name='flow_mape'),
		SMAPE(scaler=flow_scaler, pred_name='flow_pred', label_name='flow_label', name='flow_smape')
	]

	model_trainer.fit(
		begin_epoch 	= logger.best_epoch(),
		num_epochs		= args.epochs,
		train			= flow_train,
		eval			= flow_eval,
		test			= flow_test,
		metrics			= flow_metrics
	)

	net.load_parameters('%s-%04d.params' % (name, logger.best_epoch()), ctx=args.gpus)
	model_trainer.fit(
		begin_epoch		= 0,
		num_epochs		= 1,
		train			= None,
		eval			= flow_eval, 
		test			= flow_test,
		metrics			= flow_metrics
	)

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', type=str)
	parser.add_argument('--epochs', type=int)
	parser.add_argument('--gpus', type=str)
	args = parser.parse_args()

	args.gpus = [mx.gpu(int(i)) for i in args.gpus.split(',')]
	main(args)