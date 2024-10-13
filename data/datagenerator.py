### author siqing Oct 13 ###

import numpy as np
import pandas as pd

# We don't use real data
class Data(object):
	def __init__(self, length=100, sep=0.01):
		self.length = length
		self.sep = sep
  
	@staticmethod
	def BM(length=100, sep=0.01, mu=0, sigma=1):
		np.random.seed(0)
		t = np.linspace(0, length*sep, length)
		x = np.random.randn(length)
		W = np.cumsum(x) * np.sqrt(sep)
		return t*mu + sigma*W

	@staticmethod
	def OU(length=100, sep=0.01, lam=1, mu=0, sigma=1):
		np.random.seed(0)
		t = np.linspace(0, length*sep, length)
		x = np.random.randn(length)
		W = np.cumsum(x) * np.sqrt(sep)
		return mu + np.exp(-lam*t) * (W - mu) + sigma * np.exp(-2*lam*t) * np.random.randn(length)

	@staticmethod
	def GBM(length=100, sep=0.01, mu=0, sigma=1):
		np.random.seed(0)
		t = np.linspace(0, length*sep, length)
		x = np.random.randn(length)
		W = np.cumsum(x) * np.sqrt(sep)
		return np.exp((mu - 0.5*sigma**2)*t + sigma*W)

	def stock(self, **kwargs): # default GBM
		rf = kwargs.get('rf', 0.03)
		div = kwargs.get('div', 0)
		vol = kwargs.get('vol', 0.2)
		return self.GBM(self.length, mu=rf-div, sigma=vol)


if __name__ == '__main__':
	data = Data(length=20)
	print(data.stock())
	print(data.stock(rf=0.05, div=0.02, vol=0.3))
	print(data.BM())
	print(data.OU())
	print(data.GBM())
	
	