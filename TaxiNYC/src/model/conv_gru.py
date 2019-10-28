import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon import Block, HybridBlock, nn, rnn, contrib

from config import ROWS, COLUMES, FLOW_INPUT_LEN, FLOW_INPUT_DIM, FLOW_OUTPUT_DIM, FLOW_OUTPUT_LEN
from model.structure import MFDense

N_LOC = ROWS * COLUMES

class ConvGRU(Block):
	""" Convolutional Gated Recurrent Unit """
	def __init__(self, filters, hiddens, embed_dim, prefix):
		super(ConvGRU, self).__init__(prefix=prefix)
		self.filters = filters

		with self.name_scope():
			# convolutional gated recurrent units
			self.conv_grus = []
			for i, filter in enumerate(filters):
				cell = contrib.rnn.Conv2DGRUCell(
					input_shape	= (FLOW_INPUT_DIM if i == 0 else filters[i - 1],ROWS,COLUMES),
					hidden_channels	= filter,
					i2h_kernel	= (3,3),
					h2h_kernel	= (3,3),
					i2h_pad		= (1,1),
					prefix		= 'conv_gru%d_'%i					
				)
				self.register_child(cell)
				self.conv_grus += [cell]

			# dense layers (mf dense layers)
			self.denses = nn.Sequential()
			in_dims = [filters[-1]]+ hiddens
			out_dims = hiddens + [FLOW_OUTPUT_DIM * FLOW_OUTPUT_LEN]
			for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
				activation = None if i == len(in_dims) - 1 else 'relu'
				if embed_dim == 0: self.denses.add(nn.Dense(out_dim, activation, flatten=False, prefix='dense%d_'%i))
				else: self.denses.add(MFDense(N_LOC, embed_dim, in_dim, out_dim, activation, prefix='mf_dense%d_'%i))

	def forward(self, data, label):
		""" Forward process of ConvGRU.

		Parameters
		----------
		data: NDArray with shape [b, t, row, col, d].
		label: NDArray with shape [b, t, row, col, d].

		Returns
		-------
		loss: loss for gradient descent.
		(pred, label): each of them is a NDArray with shape [n, b, t, d].

		"""

		B, T, _, _, _ = data.shape
		data = nd.transpose(data, axes=(0,1,4,2,3)) # [b, t, d, row, col]

		# conv_gru
		for cell in self.conv_grus:
			data, _ = cell.unroll(length=T, inputs=data, merge_outputs=True)

		data = data[:,-1] # [b, d, row, col]
		
		data = nd.transpose(data, axes=(2,3,0,1)) # [row, col, b, d]

		data = nd.reshape(data, shape=(ROWS * COLUMES,B,-1))

		data = self.denses(data)

		data = nd.reshape(data, shape=(ROWS,COLUMES,B,FLOW_OUTPUT_LEN,-1))
		data = nd.transpose(data, axes=(2,0,1,3,4))

		label = nd.transpose(label, axes=(0,2,3,1,4)) # [b, row, col, t, d]
		label = label[:,:,:,:,:FLOW_OUTPUT_DIM]

		loss = nd.sum((data - label) ** 2)
		return loss, {'flow_pred': data, 'flow_label': label}
 
def net(settings):
	return ConvGRU(
		filters         = settings['model']['filters'],
		hiddens         = settings['model']['hiddens'],
		embed_dim		= settings['model']['embed_dim'],
		prefix          = settings['model']['type'] + "_"
	)  