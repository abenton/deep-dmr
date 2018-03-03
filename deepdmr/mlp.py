"""
Lifted from Theano multilayer perceptron tutorial.  Adapted to take the partial derivative of
some external loss w.r.t. output layer and back-propagate this loss.  Output layer has no
nonlinearity.

Adrian Benton
"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

import os
import sys
import timeit

import numpy as np
from numpy.random import RandomState

import theano
from   theano.gradient import jacobian
import theano.tensor as T

from functools import reduce

import unittest

theano.config.compute_test_value = 'off' # Use 'warn' to activate this feature, 'off' otherwise

# Test values. . .
np.random.seed(12345)
sample_n_examples = 400
sample_n_hidden = [ 50, 10, 5 ]
sample_input = np.random.randn(sample_n_examples, sample_n_hidden[0])
sample_externalGrad  = np.random.randn(sample_n_examples, sample_n_hidden[-1])

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh, includeBias=False, vname=''):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)
        
        :type n_in: int
        :param n_in: dimensionality of input
        
        :type n_out: int
        :param n_out: number of hidden units
        
        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        
        :type includeBias: bool
        :param includeBias: Whether this layer should have a bias term
        
        :type vname: str
        :param vname: name to attach to this layer's weights
        """
        
        self.input = input
        
        # Weight initialization for different nonlinearities
        if W is None:
          if activation == T.nnet.sigmoid:
            W_values = 0.01 * np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out))
                , dtype=theano.config.floatX
            )
          elif activation == T.nnet.relu:
            W_values = 0.01 * np.asarray(
                rng.normal(
                    0.0,
                    2.0/n_in,
                    size=(n_in, n_out))
                , dtype=theano.config.floatX
            )
          else:
            # Xavier initialization for tanh
            W_values = 0.01 * np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out))
                , dtype=theano.config.floatX
            )
          W = theano.shared(value=W_values.astype(theano.config.floatX),
                            name='%s_W' % (vname), borrow=True)
        
        if b is None:
          b_values = np.zeros((n_out,), dtype=theano.config.floatX)
          b = theano.shared(value=b_values.astype(theano.config.floatX),
                            name='%s_b' % (vname), borrow=True)
        
        self.W = W
        self.b = b
        
        if includeBias:
          lin_output = T.dot(input, self.W) + self.b
          self.params = [self.W, self.b]
        else:
          lin_output = T.dot(input, self.W)
          self.params = [self.W]
        
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

class MLP(object):
    """Multi-Layer Perceptron Class
    
    
    This has no softmax at the output layer, just a stack of layers with
    nonlinearities.
    """
    
    def __init__(self, seed, layerWidths, activation, outActivation, optimizer, L1_reg=0.0, L2_reg=0.0, vname=''):
        """Initialize the parameters for the multilayer perceptron
        
        :type seed: int
        :param seed: to init random number generator
        
        :type layerWidths: [ int ]
        :param layerWidths: width of each layer
        
        :type activation: Tensor -> Tensor
        :param activation: activation function for hidden layers
        
        :type outActivation: Tensor -> Tensor
        :param outActivation: activation function for output layer
        
        :type optimizer: Optimizer
        :param optimizer: generates weight updates
        
        :type L1_reg: float
        :param L1_reg: weight to place on L1 penalty on weights
        
        :type L2_reg: float
        :param L2_reg: weight to place on L2 penalty
        
        :type vname: str
        :param vname: name to attach to layer weights
        """
        
        rng = RandomState(seed)
        self.optimizer = optimizer
        
        self._dummy = T.scalar('dummy_learning_phase')
        
        self.L1_reg = theano.shared(np.array(L1_reg).astype(theano.config.floatX),
                                    'L1_%s' % (vname))
        self.L2_reg = theano.shared(np.array(L2_reg).astype(theano.config.floatX),
                                    'L2_%s' % (vname))
        
        self.__hiddenLayers = []
        self.nlayers = len(layerWidths) - 1
        
        self.inWidth  = layerWidths[0]
        self.outWidth = layerWidths[-1]
        
        self.layerActs = activation
        self.outAct    = outActivation
        
        self.input  = T.matrix('X_%s' % (vname))
        self.__externalGrad = T.matrix('ExternalGrad_%s' % (vname)) # Partial derivative of loss w.r.t. output layer -- computed somewhere else
        
        self.input.tag.test_value = sample_input
        self.__externalGrad.tag.test_value = sample_externalGrad
        
        Ws = []

        # Connect hidden layers
        for layerIndex, (nIn, nOut) in enumerate(zip(layerWidths, layerWidths[1:])):
          prevLayer = self._dummy + self.input if layerIndex == 0 else self.__hiddenLayers[-1].output
          
          act = activation if layerIndex < ( len(layerWidths) - 2 ) else outActivation
          print(str(act))
          
          hiddenLayer = HiddenLayer(
            rng=rng,
            input=prevLayer,
            n_in=nIn,
            n_out=nOut,
            activation=act,
            includeBias=True,
            vname='%s_layer-%d'  % (vname, layerIndex)
          )
          
          self.__hiddenLayers.append(hiddenLayer)
          Ws.append(hiddenLayer.W)
          
          if layerIndex == 0:
            self.L1     = abs(hiddenLayer.W).sum()
            self.L2_sqr = (hiddenLayer.W ** 2).sum()
          else:
            self.L1 += abs(hiddenLayer.W).sum()
            self.L2_sqr += (hiddenLayer.W ** 2).sum()
        
        if len(self.__hiddenLayers) > 0:
          self.output = self.__hiddenLayers[-1].output
          # L1/L2 regularization terms
          self.__reg_cost = (
            self.L1_reg * self.L1
            + self.L2_reg * self.L2_sqr
          )
          # so we can update all parameters at once
          self.__params = reduce(lambda x,y: x+y,
                               [layer.params for layer in self.__hiddenLayers])
        else:
          self.output = self.input
          self.__reg_cost = 0.0
          self.__params = []

        self.output = self.output + self._dummy
        
        # Hack to get theano autodiff to compute and backprop gradients for me.
        # Idea from Nanyun.
        self.__external_cost = T.sum( self.output * self.__externalGrad )
        
        self.__cost = self.__reg_cost + self.__external_cost
        
        # Gradient for just the external loss.
        self.__gparams = [T.grad(self.__external_cost, p) for p in self.__params]
        
        self.__reg_gparams = [T.grad(self.__reg_cost, p) for p in Ws]
        
        # Full gradient update
        self.__full_gparams = [T.grad(self.__cost, p) for p in self.__params]
        
        self.buildFns()
    
    def getWeights(self):
      wts = [p.get_value() for p in self.__params]
      return wts
    
    def setWeights(self, weights):
      '''
      Parameters
      ----------
      :type weights: [ np.array ]
      :param weights: should be the same number of elements and shapes as self.__params
      '''
      
      for param, wts in zip(self.__params, weights):
        param.set_value(np.float32(wts))
    
    def buildFns(self):
      # What to call when applying to test
      self.get_output = theano.function(
        inputs=[self.input, self._dummy],
        outputs=self.output
      )
      
      if self.__params:
        # Different cost and gradient functions for inspection/debugging.
        self.calc_gradient      = theano.function( inputs=[ self.input,
                                                            self.__externalGrad, self._dummy ],
                                                   outputs=self.__gparams )
        self.calc_regOnly_gradient = theano.function( inputs=[], outputs=self.__reg_gparams)
        self.calc_reg_gradient  = theano.function( inputs=[ self.input,
                                                            self.__externalGrad, self._dummy ],
                                                   outputs=self.__full_gparams )
        self.calc_external_cost = theano.function( inputs=[ self.input,
                                                            self.__externalGrad, self._dummy ],
                                                   outputs=self.__external_cost )
        self.calc_reg_cost      = theano.function( inputs=[ ], outputs=self.__reg_cost )
        self.calc_total_cost    = theano.function( inputs=[ self.input,
                                                            self.__externalGrad,
                                                            self._dummy ],
                                                   outputs=self.__cost )
        
        # For debugging, get hidden layer values
        self.get_layer_values = theano.function(inputs=[ self.input, self._dummy ],
                                                outputs=[h.output for h in self.__hiddenLayers]
         )
      else:
        self.calc_gradient = lambda x,y: []
        self.calc_regOnly_gradient = lambda: []
        self.calc_reg_gradient = lambda x,y: []
        self.calc_external_cost = lambda x,y: 0.0
        self.calc_reg_cost = lambda: 0.0
        self.calc_total_cost = lambda x,y: 0.0
      
      if self.optimizer is not None:
        self.setOptimizer(self.optimizer)
    
    def setOptimizer(self, optimizer):
      if self.__params:
        self.optimizer = optimizer
        self.__updates = self.optimizer.getUpdates(self.__params, self.__full_gparams)
        
        self.take_step = theano.function([ self.input, self.__externalGrad, self._dummy ],
                                         outputs=[],
                                         updates=self.__updates)
      else:
        self.take_step = lambda x: None
    
    def getParams(self):
      return self.__params

    def getRegCost(self):
      return self.__reg_cost
