import unittest
import numpy as num
from activation import Tanh
from perceptron import MultiPerceptron

def bias_add(instances):
  amountOfInstances = instances.shape[0]
  bias = -num.ones( (amountOfInstances,1) )
  return num.concatenate( ( instances, bias ), axis=1)

def zeroInitializer(numberOfRows,numberOfColumns): 
  return num.zeros( (numberOfRows,numberOfColumns) ) 

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
		self.assertEqual( model._layerOutputs[0].shape, (1,4) ) 

		self.assertArrayEqual( model._weights[0], num.array( [[0,0], [0,0], [0,0], [0,0]] ) )
		self.assertArrayEqual( model._layerOutputs[0], [[0,0,0,0]] ) 

		model.addLayer( 2, 1 )
		
		self.assertEqual( len(model._weights), 2 )
		self.assertEqual( len(model._layerOutputs), 2 )

		self.assertEqual( model._weights[0].shape, (4,2) )
		self.assertEqual( model._weights[1].shape, (3,1) )
		self.assertEqual( model._layerOutputs[0].shape, (1,4) ) 
		self.assertEqual( model._layerOutputs[1].shape, (1,3) ) 

		self.assertArrayEqual( model._weights[0], num.array( [[0,0], [0,0], [0,0], [0,0]] ) )
		self.assertArrayEqual( model._weights[1], num.array( [[0], [0], [0]] ) )
		self.assertArrayEqual( model._layerOutputs[0], [[0,0,0,0]] ) 
		self.assertArrayEqual( model._layerOutputs[1], [[0,0,0]] ) 

	def test_addLayersAtOnce(self):
		model = MultiPerceptron( activation=self.tanh, weightsInitializer=zeroInitializer )
		model.configureLayers( [3, 2, 1] )

		self.assertEqual( len(model._weights), 2 )
		self.assertEqual( len(model._layerOutputs), 3 )

		self.assertEqual( model._weights[0].shape, (4,2) )
		self.assertEqual( model._weights[1].shape, (3,1) )
		self.assertEqual( model._layerOutputs[0].shape, (1,4) ) 
		self.assertEqual( model._layerOutputs[1].shape, (1,3) ) 
		self.assertEqual( model._layerOutputs[2].shape, (1,1) ) 

		self.assertArrayEqual( model._weights[0], num.array( [[0,0], [0,0], [0,0], [0,0]] ) )
		self.assertArrayEqual( model._weights[1], num.array( [[0], [0], [0]] ) )
		self.assertArrayEqual( model._layerOutputs[0], [[0,0,0,0]] ) 
		self.assertArrayEqual( model._layerOutputs[1], [[0,0,0]] ) 
		self.assertArrayEqual( model._layerOutputs[2], [[0]] ) 

	def test_propagateForward(self):
		S = [ 3,2,1 ]
		Xh = num.array([[1,0,1]])
		
		model = MultiPerceptron( activation=self.tanh, weightsInitializer=zeroInitializer )
		model.configureLayers( S )
		model.propagateForward( Xh )
		
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

if __name__ == '__main__':
	unittest.main()
