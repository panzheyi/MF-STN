import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon import Block, HybridBlock, nn, rnn

from config import ROWS, COLUMES, FLOW_INPUT_LEN, FLOW_INPUT_DIM, FLOW_OUTPUT_DIM, FLOW_OUTPUT_LEN
from model.structure import MFDense

N_LOC = ROWS * COLUMES

class GRU(Block):
	""" Gated Recurrent Unit """
	def __init__(self, rnn_hiddens, hiddens, embed_dim, prefix):
		super(GRU, self).__init__(prefix=prefix)

		with self.name_scope():
			# gated recurrent units
			self.grus = []
			for i, hidden in enumerate(rnn_hiddens):
				cell = rnn.GRUCell(hidden, prefix='gru%d_'%i)
				self.register_child(cell)
				self.grus += [cell]

			# dense layers (mf dense layers)
			self.denses = nn.Sequential()
			in_dims = [rnn_hiddens[-1]]+ hiddens
			out_dims = hiddens + [FLOW_OUTPUT_DIM * FLOW_OUTPUT_LEN]
			for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
				activation = None if i == len(in_dims) - 1 else 'relu'
				if embed_dim == 0: self.denses.add(nn.Dense(out_dim, activation, flatten=False, prefix='dense%d_'%i))
				else: self.denses.add(MFDense(N_LOC, embed_dim, in_dim, out_dim, activation, prefix='mf_dense%d_'%i))

	def forward(self, data, label):
		""" Forward process of GRU.

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

		data = nd.transpose(data, axes=(0,2,3,1,4)) # [b, row, col, t, d]

		data = nd.reshape(data, shape=(B * ROWS * COLUMES,T,-1))

		# gru
		for cell in self.grus:
			data, _ = cell.unroll(length=T, inputs=data, merge_outputs=True)
		
		data = data[:,-1]

		data = nd.reshape(data, shape=(B,ROWS * COLUMES,-1))
		data = nd.transpose(data, axes=(1, 0, 2)) # [row * col, b, d]

		data = self.denses(data)

		data = nd.reshape(data, shape=(ROWS,COLUMES,B,FLOW_OUTPUT_LEN,-1))
		data = nd.transpose(data, axes=(2,0,1,3,4))

		label = nd.transpose(label, axes=(0,2,3,1,4)) # [b, row, col, t, d]
		label = label[:,:,:,:,:FLOW_OUTPUT_DIM]

		loss = nd.sum((data - label) ** 2)
		return loss, {'flow_pred': data, 'flow_label': label}
 
def net(settings):
	return GRU(
		rnn_hiddens     = settings['model']['rnn_hiddens'],
		hiddens         = settings['model']['hiddens'],
		embed_dim   	= settings['model']['embed_dim'],
		prefix          = settings['model']['type'] + "_"
	)  