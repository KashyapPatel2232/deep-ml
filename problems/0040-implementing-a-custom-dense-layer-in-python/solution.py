import numpy as np
import copy
import math

# DO NOT CHANGE SEED
np.random.seed(42)

# DO NOT CHANGE LAYER CLASS
class Layer(object):

	def set_input_shape(self, shape):
		self.input_shape = shape

	def layer_name(self):
		return self.__class__.__name__

	def parameters(self):
		return 0

	def forward_pass(self, X, training):
		raise NotImplementedError()

	def backward_pass(self, accum_grad):
		raise NotImplementedError()

	def output_shape(self):
		raise NotImplementedError()

# Your task is to implement the Dense class based on the above structure
class Dense(Layer):
	def __init__(self, n_units, input_shape=None):
		self.layer_input = None
		self.input_shape = input_shape
		self.n_units = n_units
		self.trainable = True
		self.W = None
		self.w0 = None

	def initialize(self, optimizer):
		# Initialize weights W, biases w0, and optimizers
		self.W = np.random.uniform(low = -1/(np.sqrt(self.input_shape[0])), high = 1/(np.sqrt(self.input_shape[0])), size=[self.input_shape[0], self.n_units])
		self.w0 = np.zeros(self.n_units) 
		self.opt_W = optimizer
		self.opt_w0 = optimizer

	def parameters(self):
		# Return total number of parameters
		return self.input_shape[0]*self.n_units + self.n_units

	def forward_pass(self, X, training=True):
		# Compute and return the forward pass
		self.layer_input = X
		return  X @ self.W + self.w0

	def backward_pass(self, accum_grad):
		# Compute gradients, update weights if trainable, return grad w.r.t. input
		grad_W = np.outer(self.layer_input, accum_grad)
		grad_w0 = accum_grad
		grad_X = accum_grad @ self.W.T
		if self.trainable == True:
			self.opt_W.update(self.W, grad_W)
			self.opt_w0.update(self.w0, grad_w0)
		return grad_X

	def output_shape(self):
		# Return output shape tuple
		return (self.n_units,)