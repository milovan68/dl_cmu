import torch
import torch.nn as nn
import numpy as np
import itertools


class Sigmoid:

	def __init__(self):
		pass
	def forward(self, x):
		self.res = 1/(1+np.exp(-x))
		return self.res
	def backward(self):
		return self.res * (1-self.res)
	def __call__(self, x):
		return self.forward(x)


class Tanh:
	def __init__(self):
		pass
	def forward(self, x):
		self.res = np.tanh(x)
		return self.res
	def backward(self):
		return 1 - (self.res**2)
	def __call__(self, x):
		return self.forward(x)


class GRU_Cell:
	"""docstring for GRU_Cell"""
	def __init__(self, in_dim, hidden_dim):
		self.d = in_dim
		self.h = hidden_dim
		h = self.h
		d = self.d

		self.Wzh = np.random.randn(h,h)
		self.Wrh = np.random.randn(h,h)
		self.Wh  = np.random.randn(h,h)

		self.Wzx = np.random.randn(h,d)
		self.Wrx = np.random.randn(h,d)
		self.Wx  = np.random.randn(h,d)



		self.dWzh = np.zeros((h,h))
		self.dWrh = np.zeros((h,h))
		self.dWh  = np.zeros((h,h))

		self.dWzx = np.zeros((h,d))
		self.dWrx = np.zeros((h,d))
		self.dWx  = np.zeros((h,d))

		self.z_act = Sigmoid()
		self.r_act = Sigmoid()
		self.h_act = Tanh()

		
	def forward(self, x, h):
		# input:
		# 	- x: shape(input dim),  observation at current time-step
		# 	- h: shape(hidden dim), hidden-state at previous time-step
		# 
		# output:
		# 	- h_t: hidden state at current time-step
		self.x = x
		self.h = h
		z_t = self.z_act.forward(np.dot(self.Wzh, h) + np.dot(self.Wzx, x))
		self.z_t = z_t
		r_t = self.r_act.forward(np.dot(self.Wrh, h) + np.dot(self.Wrx, x))
		self.r_t = r_t
		self.a9 = r_t * h
		h_hat_t = self.h_act.forward(np.dot(self.Wh, np.multiply(r_t, h)) + np.dot(self.Wx, x))
		self.h_hat_t = h_hat_t
		h_t = np.multiply((1 - z_t), h) + np.multiply(z_t, h_hat_t)
		return h_t



	def backward(self, delta):
		self.dWzh = np.dot(self.h.reshape(self.h.shape[0], -1), (delta * self.h_hat_t.T - delta * self.h.T) * self.z_act.backward().T).T
		print(self.dWzh.shape, self.x.shape, self.h.shape)
		self.dWrh = np.dot(self.h.reshape(self.h.shape[0], -1), np.dot(delta * self.z_t.T * self.h_act.backward().T, self.Wh) * self.h.T * self.r_act.backward().T).T
		print(self.dWrh.shape)
		self.dWh = np.dot(self.a9.reshape(self.a9.shape[0], -1), delta * self.z_t.T * self.h_act.backward().T).T
		print(self.dWh.shape)
		self.dWzx = np.dot(self.x.reshape(self.x.shape[0], -1), (delta * self.h_hat_t.T - delta * self.h.T) * self.z_act.backward().T).T
		print(self.dWzx.shape)
		self.dWrx = np.dot(self.x.reshape(self.x.shape[0], -1), np.dot(delta * self.z_t.T * self.h_act.backward().T, self.Wh) * self.h.T * self.r_act.backward().T).T
		print(self.dWrx.shape)
		self.dWx = np.dot(self.x.reshape(self.x.shape[0], -1), delta * self.z_t.T * self.h_act.backward().T).T
		print(self.dWx.shape)
		dx = np.dot((delta * self.h_hat_t.T - delta * self.h.T) * self.z_act.backward().T, self.Wzx) + np.dot(np.dot(delta * self.z_t.T * self.h_act.backward().T, self.Wh) * self.h.T * self.r_act.backward().T, self.Wrx) + np.dot(delta * self.z_t.T * self.h_act.backward().T, self.Wx)
		dh = np.dot((delta * self.h_hat_t.T - delta * self.h.T) * self.z_act.backward().T, self.Wzh) + np.dot(
			np.dot(delta * self.z_t.T * self.h_act.backward().T, self.Wh) * self.h.T * self.r_act.backward().T,
			self.Wrh) + np.dot(delta * self.z_t.T * self.h_act.backward().T, self.Wh) * self.r_t.T + delta * (
						 1 - self.z_t).T
		return dx, dh











if __name__ == '__main__':
	test()









