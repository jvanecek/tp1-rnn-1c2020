import numpy as num
from perceptron_tools import RandomInitializer
from activation import ActivationParser

class MultiPerceptron():

	def _addBias(self, instances):
		if self._useBias:
			amountOfInstances = instances.shape[0]
			bias = -num.ones( (amountOfInstances,1) )
			return num.concatenate( ( instances, bias ), axis=1)
		else:
			return instances

	def _subBias(self, instances):
		if self._useBias:
			return instances[:,:-1]
		else:
			return instances

	def __init__(self, activation, weightsInitializer, useBias=True):
		self._activation = activation
		self._weightsInitializer = weightsInitializer
		self._useBias = useBias
		self._weights = []
		self._layerOutputs = []

	def addLayer(self, inputFeatures, outputUnits):
		inputPlusBias = inputFeatures+self._useBias
		Wi = self._weightsInitializer(inputPlusBias, outputUnits)

		self._weights.append( Wi )
		self._layerOutputs.append( None )


	def configureLayers(self, unitsPerLayer):
		for i in range(len(unitsPerLayer)-1):
			self.addLayer( unitsPerLayer[i], unitsPerLayer[i+1] )

		self._layerOutputs.append( None )


	def propagateForward(self, input):
		for i in range(len(self._layerOutputs)):
			if i == 0:
				Yi = self._addBias( input )

			elif i < len(self._layerOutputs)-1:
				preActivation = num.dot( self._layerOutputs[i-1], self._weights[i-1] )
				Yi = self._addBias( self._activation( preActivation ) )

			else:
				preActivation = num.dot( self._layerOutputs[i-1], self._weights[i-1] )
				Yi = self._activation( preActivation )

			self._layerOutputs[i] = Yi
		return self._layerOutputs[-1]


	def propagateBackwards(self, expectedOutput, learningRate, updateWeights=True):
		E = None
		dWs = [None]*len(self._weights)
		Ds = [None]*len(self._layerOutputs)

		for i in reversed(range(len(self._weights))):

			dWs[i] = num.zeros_like( self._weights[i] )
			dY = self._activation.derivative( self._layerOutputs[i+1] )

			if i == len(self._weights)-1:
				E = Ei = expectedOutput - self._layerOutputs[-1]
				Di = Ei*dY
			else:
				Ei = num.dot( Ds[i+2], self._weights[i+1].T )
				Di = self._subBias( Ei*dY )

			dWs[i] = learningRate * num.dot( self._layerOutputs[i].T, Di )
			Ds[i+1] = Di

		if updateWeights:
			for i in range(len(self._weights)):
				self._weights[i] += dWs[i]

		return E

	def summary(self):
		summary = ''
		summary += 'Activation: {}\n'.format(self._activation.description())
		summary += 'With Bias: {}\n'.format(self._useBias)
		summary += 'Layers: {}\n'.format(len(self._weights))

		shapes = [w.shape for w in self._weights]
		summary += 'Weights: {}\n'.format( shapes )
		params = num.sum( [ w[0]*w[1] for w in shapes ] )
		summary += 'Trainable params: {}'.format(params)

		return summary

class MultiPerceptronParser():
	def parse(self, paramsDict, inputFeatures):
		outputUnits = 1
		activation = ActivationParser().parse( paramsDict['activation'] )
		mean = float(paramsDict['weightMean'])
		stdv = float(paramsDict['weightStdv'])

		hideLayersUnits = [int(unit) for unit in paramsDict['hiddenLayersUnits'].split(',')]
		unitsPerLayer = [inputFeatures]+hideLayersUnits+[outputUnits]

		model = MultiPerceptron( activation=activation, weightsInitializer=RandomInitializer(mean, stdv) )
		model.configureLayers( unitsPerLayer )

		return model