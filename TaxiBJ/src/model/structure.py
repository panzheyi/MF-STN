import mxnet as mx
from mxnet import nd
from mxnet.gluon import Block, HybridBlock, nn, rnn

class MFDense(Block):
	""" Maxtrix factorization based dense layer"""
	def __init__(self, num_node, embed_dim, in_dim, out_dim, activation=None, **kwargs):
		super(MFDense, self).__init__(**kwargs)

		self.in_dim = in_dim
		self.out_dim = out_dim
		self.activation = activation

		with self.name_scope():
			self.w1 = self.params.get('w1_weight', shape=(num_node,embed_dim), allow_deferred_init=True)
			self.w2 = self.params.get('w2_weight', shape=(embed_dim, in_dim * out_dim), allow_deferred_init=True)
			self.b1 = self.params.get('b1_weight', shape=(num_node, embed_dim), allow_deferred_init=True)
			self.b2 = self.params.get('b2_weight', shape=(embed_dim, out_dim), allow_deferred_init=True)

	def forward(self, data):
		""" Forward process of MFDense layer

		Parameters
		----------
		data: NDArray with shape [n, b, in_dim]

		Returns
		-------
		output: NDArray with shape [n, b, out_dim]
		"""
		ctx = data.context
		weight = nd.dot(self.w1.data(ctx), self.w2.data(ctx)).reshape((-1,self.in_dim,self.out_dim)) # [n, in_dim, out_dim]
		bias = nd.dot(self.b1.data(ctx), self.b2.data(ctx)).reshape((-1,1,self.out_dim))  # [n, 1, out_dim]
		output = nd.batch_dot(data, weight) + bias
		if self.activation is not None: output = nd.Activation(output, act_type=self.activation)
		return output

class ResUnit(HybridBlock):
	""" Residual unit"""
	def __init__(self, filter, resample, prefix=None):
		super(ResUnit, self).__init__(prefix=prefix)
		self.resample = resample
		with self.name_scope():
			self.bn1 = nn.BatchNorm()
			self.bn2 = nn.BatchNorm()
			self.conv1 = nn.Conv2D(filter, kernel_size=3, strides=1, padding=1)
			self.conv2 = nn.Conv2D(filter, kernel_size=3, strides=1, padding=1)
			if self.resample: self.conv_resample = nn.Conv2D(filter, kernel_size=1, strides=1, padding=0)

	def hybrid_forward(self, F, x):
		residual = x
		x = self.bn1(x)
		x = F.Activation(x, act_type='relu')
		x = self.conv1(x)

		x = self.bn2(x)
		x = F.Activation(x, act_type='relu')
		x = self.conv2(x)
		
		if self.resample: residual = self.conv_resample(residual)
		return x + residual