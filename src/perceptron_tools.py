import numpy as num

class WeightsInitializer():
	def __call__(self, numberOfRows,numberOfColumns):
		raise Exception('This method should be implemented by subclass')

class ZeroInitializer():
	def __call__(self, numberOfRows,numberOfColumns):
		return num.zeros( (numberOfRows,numberOfColumns) )

class RandomInitializer():
	def __init__(self, mean=0, stdv=1):
		self._mean = 0
		self._stdv = 1

	def __call__(self, numberOfRows,numberOfColumns):
		return num.random.normal( self._mean, self._stdv, (numberOfRows,numberOfColumns) )
