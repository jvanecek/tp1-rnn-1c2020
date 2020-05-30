
import numpy as num

class ActivationFunction():

	def _subclassResponsibility(self):
		raise Exception('This method should be implemented by subclass')

	def __call__(self, X):
		 self._subclassResponsibility()

	def description(self):
		self._subclassResponsibility()

	def derivative(self):
		self._subclassResponsibility()


class Sigmoid(ActivationFunction):
	def __call__(self, X):
		return 1 / (1 + num.exp(-X)) 

	def description(self):
		return 'Sigmoid'

	def derivative(self, Y):
		return Y*(1-Y)


class Tanh(ActivationFunction):
	def __call__(self, X):
		return num.tanh(X) 

	def description(self):
		return 'Tanh'

	def derivative(self, Y):
		return 1-Y**2


class Sign(ActivationFunction):
	def __call__(self, X):
		return num.sign(X) 

	def description(self):
		return 'Sign'

	def derivative(self, Y):
		return 0