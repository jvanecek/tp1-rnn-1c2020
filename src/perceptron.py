import numpy as num

def bias_add(instances, useBias):
	if useBias:
		amountOfInstances = instances.shape[0]
		bias = -num.ones( (amountOfInstances,1) )
		return num.concatenate( ( instances, bias ), axis=1)
	else:
		return instances

def bias_sub(instances, useBias): 
	if useBias: 
		return instances[:,:-1]
	else: 
		return instances

class MultiPerceptron():

	def __init__(self, activation, weightsInitializer, useBias=True):
		self._activation = activation
		self._weightsInitializer = weightsInitializer
		self._useBias = useBias 
		self._weights = []
		self._layerOutputs = []

	def addLayer(self, inputFeatures, outputUnits):
		inputPlusBias = inputFeatures+self._useBias
		Wi = self._weightsInitializer(inputPlusBias, outputUnits)
		Yi = num.zeros( (1, inputPlusBias) )

		self._weights.append( Wi )
		self._layerOutputs.append( Yi )


	def configureLayers(self, unitsPerLayer):
		for i in range(len(unitsPerLayer)-1):
			self.addLayer( unitsPerLayer[i], unitsPerLayer[i+1] )

		Ylast = num.zeros( (1, unitsPerLayer[-1]) )
		self._layerOutputs.append( Ylast )


	def propagateForward(self, input):
		for i in range(len(self._layerOutputs)):
			if i == 0:
				self._layerOutputs[i][:] = bias_add( input, self._useBias )
			
			elif i < len(self._layerOutputs)-1:
				preActivation = num.dot( self._layerOutputs[i-1], self._weights[i-1] )
				self._layerOutputs[i][:] = bias_add( self._activation( preActivation ), self._useBias )

			else:
				preActivation = num.dot( self._layerOutputs[i-1], self._weights[i-1] )
				self._layerOutputs[i][:] = self._activation( preActivation )
				