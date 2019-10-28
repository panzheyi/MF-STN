import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon import Block, HybridBlock, nn, rnn

from config import ROWS, COLUMES, FLOW_OUTPUT_DIM, FLOW_OUTPUT_LEN
from model.structure import MFDense, ResUnit

N_LOC = ROWS * COLUMES

class CNN(Block):
    """ Convolutional neural network """
    def __init__(self, filters, hiddens, embed_dim, prefix):
        super(CNN, self).__init__(prefix=prefix)
        self.filters = filters

        with self.name_scope():
            # convolutional layers
            self.convs = nn.Sequential()
            for i, filter in enumerate(filters):
                self.convs.add(nn.Conv2D(filter, kernel_size=3, strides=1, padding=1, activation='relu', prefix='cnn%d_'%i))

            # dense layers (mf dense layers)
            self.denses = nn.Sequential()
            in_dims = [filters[-1]]+ hiddens
            out_dims = hiddens + [FLOW_OUTPUT_DIM * FLOW_OUTPUT_LEN]
            for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
                activation = None if i == len(in_dims) - 1 else 'relu'
                if embed_dim == 0: self.denses.add(nn.Dense(out_dim, activation, flatten=False, prefix='dense%d_'%i))
                else: self.denses.add(MFDense(N_LOC, embed_dim, in_dim, out_dim, activation, prefix='mf_dense%d_'%i))

    def forward(self, data, label):
        """ Forward process of CNN.

        Parameters
        ----------
        data: NDArray with shape [b, t, row, col, d].
        label: NDArray with shape [b, t, row, col, d].

        Returns
        -------
        loss: loss for gradient descent.
        (pred, label): each of them is a NDArray with shape [n, b, t, d].
        
        """
        B = data.shape[0]
        data = nd.transpose(data, axes=(0,1,4,2,3)) # [b, t, d, row, col]
        data = nd.reshape(data, shape=(B,-1,ROWS,COLUMES)) # [b, t * d, row, col]

        # convolution layers
        data = self.convs(data)
        data = nd.transpose(data, axes=(2,3,0,1)) # [row, col, b, d]
        data = nd.reshape(data, shape=(ROWS * COLUMES,B,-1))

        # dense layers
        data = self.denses(data)
        data = nd.reshape(data, shape=(ROWS,COLUMES,B,FLOW_OUTPUT_LEN,-1))
        data = nd.transpose(data, axes=(2,0,1,3,4))

        label = nd.transpose(label, axes=(0,2,3,1,4)) # [b, row, col, t, d]
        label = label[:,:,:,:,:FLOW_OUTPUT_DIM]

        loss = nd.sum((data - label) ** 2)
        return loss, {'flow_pred': data, 'flow_label': label}
 
def net(settings):
    return CNN(
        filters         = settings['model']['filters'],
        hiddens         = settings['model']['hiddens'],
        embed_dim       = settings['model']['embed_dim'],
        prefix          = settings['model']['type'] + "_"
    )  