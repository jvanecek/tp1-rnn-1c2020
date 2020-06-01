import unittest
import numpy as num
from activation import Tanh
from perceptron import MultiPerceptron, MultiPerceptronParser
from perceptron_tools import ZeroInitializer

def bias_add(instances):
	amountOfInstances = instances.shape[0]
	bias = -num.ones( (amountOfInstances,1) )
	return num.concatenate( ( instances, bias ), axis=1)

def bias_sub(instances):
	return instances[:,:-1]

zeroInitializer = ZeroInitializer()

class TestMultiPerceptron(unittest.TestCase):
	tanh = Tanh()

	def assertArrayEqual(self, array1, array2):
		num.testing.assert_array_equal(array1, array2)

	def test_initialization(self):
		model = MultiPerceptron( activation=self.tanh, weightsInitializer=zeroInitializer )

		self.assertEqual( model._activation, self.tanh )
		self.assertEqual( model._weightsInitializer, zeroInitializer )
		self.assertTrue( model._useBias )
		self.assertEqual( len(model._weights), 0 )
		self.assertEqual( len(model._layerOutputs), 0 )

	def test_addLayers(self):
		model = MultiPerceptron( activation=self.tanh, weightsInitializer=zeroInitializer )
		model.addLayer( 3, 2 )

		self.assertEqual( len(model._weights), 1 )
		self.assertEqual( len(model._layerOutputs), 1 )

		self.assertEqual( model._weights[0].shape, (4,2) )

		self.assertArrayEqual( model._weights[0], num.array( [[0,0], [0,0], [0,0], [0,0]] ) )
		self.assertArrayEqual( model._layerOutputs[0], [None] )

		model.addLayer( 2, 1 )

		self.assertEqual( len(model._weights), 2 )
		self.assertEqual( len(model._layerOutputs), 2 )

		self.assertEqual( model._weights[0].shape, (4,2) )
		self.assertEqual( model._weights[1].shape, (3,1) )

		self.assertArrayEqual( model._weights[0], num.array( [[0,0], [0,0], [0,0], [0,0]] ) )
		self.assertArrayEqual( model._weights[1], num.array( [[0], [0], [0]] ) )
		self.assertArrayEqual( model._layerOutputs[0], [None] )
		self.assertArrayEqual( model._layerOutputs[1], [None] )

	def test_addLayersAtOnce(self):
		model = MultiPerceptron( activation=self.tanh, weightsInitializer=zeroInitializer )
		model.configureLayers( [3, 2, 1] )

		self.assertEqual( len(model._weights), 2 )
		self.assertEqual( len(model._layerOutputs), 3 )

		self.assertEqual( model._weights[0].shape, (4,2) )
		self.assertEqual( model._weights[1].shape, (3,1) )

		self.assertArrayEqual( model._weights[0], num.array( [[0,0], [0,0], [0,0], [0,0]] ) )
		self.assertArrayEqual( model._weights[1], num.array( [[0], [0], [0]] ) )
		self.assertArrayEqual( model._layerOutputs[0], [None] )
		self.assertArrayEqual( model._layerOutputs[1], [None] )
		self.assertArrayEqual( model._layerOutputs[2], [None] )

	def test_propagateForward(self):
		S = [ 3,2,1 ]
		Xh = num.array([[1,0,1]])

		model = MultiPerceptron( activation=self.tanh, weightsInitializer=zeroInitializer )
		model.configureLayers( S )
		Y = model.propagateForward( Xh )

		W1 = zeroInitializer(S[0]+1, S[1])
		W2 = zeroInitializer(S[1]+1, S[2])
		Y0 = num.zeros( (1,S[0]+1) )
		Y1 = num.zeros( (1,S[1]+1) )
		Y2 = num.zeros( (1,S[2]) )

		Y0[:] = bias_add( Xh )
		Y1[:] = bias_add( num.tanh( num.dot( Y0, W1)) )
		Y2[:] = num.tanh( num.dot( Y1, W2))

		self.assertArrayEqual( model._layerOutputs[0], Y0 )
		self.assertArrayEqual( model._layerOutputs[1], Y1 )
		self.assertArrayEqual( model._layerOutputs[2], Y2 )
		self.assertArrayEqual( Y, Y2 )

	def test_propagateBackwards(self):
		S = [ 3,2,1 ]
		Xh = num.array([[1,0,1]])
		Zh = num.array([[1]])
		lr = 0.1

		model = MultiPerceptron( activation=self.tanh, weightsInitializer=zeroInitializer )
		model.configureLayers( S )
		model.propagateForward( Xh )

		W1 = zeroInitializer(S[0]+1, S[1])
		W2 = zeroInitializer(S[1]+1, S[2])

		self.assertArrayEqual( model._weights[0], W1 )
		self.assertArrayEqual( model._weights[1], W2 )

		E = model.propagateBackwards( Zh, lr )

		Y0 = num.zeros( (1,S[0]+1) )
		Y1 = num.zeros( (1,S[1]+1) )
		Y2 = num.zeros( (1,S[2]) )

		Y0[:] = bias_add( Xh )
		Y1[:] = bias_add( num.tanh( num.dot( Y0, W1)) )
		Y2[:] = num.tanh( num.dot( Y1, W2))

		dW1= num.zeros_like( W1 )
		dW2 = num.zeros_like( W2 )

		E2 = Zh-Y2
		dY2 = 1-num.square(Y2)
		D2 = E2*dY2
		dW2 += lr * num.dot( Y1.T, D2)

		E1 = num.dot( D2, W2.T )
		dY1 = 1-num.square(Y1)
		D1 = bias_sub( E1*dY1 )
		dW1 += lr * num.dot( Y0.T, D1)

		W1 += dW1
		W2 += dW2

		self.assertArrayEqual( E, E2 )
		self.assertArrayEqual( model._layerOutputs[0], Y0 )
		self.assertArrayEqual( model._layerOutputs[1], Y1 )
		self.assertArrayEqual( model._layerOutputs[2], Y2 )
		self.assertArrayEqual( model._weights[0], W1 )
		self.assertArrayEqual( model._weights[1], W2 )

	def test_summary(self):
		model = MultiPerceptron( activation=self.tanh, weightsInitializer=zeroInitializer )
		model.configureLayers( [2,3,1] )

		expectedSummary = """Activation: Tanh
With Bias: True
Layers: 2
Weights: [(3, 3), (4, 1)]
Trainable params: 13"""

		self.assertEqual( model.summary(), expectedSummary )

class TestMultiPerceptronParser(unittest.TestCase):
	def test_parseWithBias(self):
		parser = MultiPerceptronParser()
		params = {
			'activation' : 'Tanh',
			'weightMean' : '0',
			'weightStdv' : '0.5',
			'hiddenLayersUnits' : '2,1'
		}

		model = parser.parse(params, 10)
		expectedSummary = """Activation: Tanh
With Bias: True
Layers: 3
Weights: [(11, 2), (3, 1), (2, 1)]
Trainable params: 27"""

		self.assertEqual( model.summary(), expectedSummary )

if __name__ == '__main__':
	unittest.main()