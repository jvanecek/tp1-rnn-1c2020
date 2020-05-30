import numpy as num
import unittest
from activation import Sigmoid, Tanh, Sign

class TestSigmoid(unittest.TestCase):

	def test_description(self):
		sigmoid = Sigmoid()

		self.assertEqual(sigmoid.description(), 'Sigmoid')

	def test_evaluateOnNumbers(self):
		sigmoid = Sigmoid()

		self.assertEqual(sigmoid(1), 1 / (1 + num.exp(-1)))
		self.assertEqual(sigmoid(0.73), 1 / (1 + num.exp(-0.73)))

	def test_evaluateOnVectors(self):
		sigmoid = Sigmoid()

		vector = num.array([1,2,3])
		num.testing.assert_array_equal(sigmoid(vector), 1 / (1 + num.exp(-vector)))
	
	def test_evaluateDerivativeOnNumbers(self):
		sigmoid = Sigmoid()

		self.assertEqual(sigmoid.derivative(2), 2*(1-2))

	def test_evaluateDerivativeOnVectors(self):
		sigmoid = Sigmoid()

		vector = num.array([1,2,3])
		num.testing.assert_array_equal(sigmoid.derivative(vector), num.array([0,-2,-6]))


class TestTanh(unittest.TestCase):

	def test_description(self):
		tanh = Tanh()

		self.assertEqual(tanh.description(), 'Tanh')

	def test_evaluateOnNumbers(self):
		tanh = Tanh()

		self.assertEqual(tanh(1), num.tanh(1))
		self.assertEqual(tanh(0.75), num.tanh(0.75))

	def test_evaluateOnVectors(self):
		tanh = Tanh()

		vector = num.array([1,2,3])
		num.testing.assert_array_equal(tanh(vector), num.tanh(vector))
	
	def test_evaluateDerivativeOnNumbers(self):
		tanh = Tanh()

		self.assertEqual(tanh.derivative(1), 0)
		self.assertEqual(tanh.derivative(2), -3)

	def test_evaluateDerivativeOnVectors(self):
		tanh = Tanh()

		vector = num.array([1,2,3])
		num.testing.assert_array_equal(tanh.derivative(vector), num.array([0,-3,-8]))


class TestSign(unittest.TestCase):

	def test_description(self):
		sign = Sign()

		self.assertEqual(sign.description(), 'Sign')

	def test_evaluateOnNumbers(self):
		sign = Sign()

		self.assertEqual(sign(-2.5), -1)
		self.assertEqual(sign(-1.3), -1)
		self.assertEqual(sign(0), 0)
		self.assertEqual(sign(0.75), 1)
		self.assertEqual(sign(2.8), 1)

	def test_evaluateOnVectors(self):
		sign = Sign()

		vector = num.array([-13.3,0,3])
		num.testing.assert_array_equal(sign(vector), num.array([-1,0,1]))
	
	def test_evaluateDerivativeOnNumbers(self):
		sign = Sign()

		# No estoy muy seguro. Revisar
		self.assertEqual(sign.derivative(-1.5), 0)
		self.assertEqual(sign.derivative(0), 0)
		self.assertEqual(sign.derivative(3.5), 0)

	def test_evaluateDerivativeOnVectors(self):
		sign = Sign()

		vector = num.array([-5,10,0])
		num.testing.assert_array_equal(sign.derivative(vector), num.array([0,0,0]))


if __name__ == '__main__':
	unittest.main()
