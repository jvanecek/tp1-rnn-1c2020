import math
import numpy as num
import time
from numpy import linalg as LA
import logging

def meanSquared(estimation):
	return num.mean( num.sum( num.square( estimation ), axis=1) )

class TrainingLogger():
	def logStart(self, training, model, input, validationSet):
		log = "Starting {}\n".format(training.description())
		log += "> Training instances: {}\n".format(input.shape[0])
		if validationSet is not None:
			log += "> Validation instances: {}\n".format(validationSet.shape[0])
		log += "> Characteristics of model to train\n{}".format(model.summary())
		logging.info(log)
		self._start_time = time.time()

	def logEnd(self, training, lastEpoch, validationLoss):
		executionTime = time.time()-self._start_time
		log = "Finished {}\n".format(training.description())
		log += "> Epochs: {}\n".format(lastEpoch)
		log += "> Training Loss: {:.4f}\n".format(training.lastLoss())
		if validationLoss is not None:
			log += "> Validation Loss: {:.4f}\n".format(validationLoss)
		log += "> Time: {:.4f} seconds".format(executionTime)
		logging.info(log)

class TrainingScenario():
	def _subclassResponsibility(self):
		raise Exception('This method should be implemented by subclass')

	def description(self):
		self._subclassResponsibility

	def executeOn(self,
			model,
			input,
			target,
			learningRate,
			stopCondition,
			validationSet=None,
			validationTarget=None):
		self._subclassResponsibility()

	def historicalLoss(self):
		self._subclassResponsibility()

	def lastLoss(self):
		self._subclassResponsibility()

	def _fit(self, model, input, target, learningRate):
		model.propagateForward( input )
		estimation = model.propagateBackwards( target, learningRate )
		return meanSquared( estimation )

	def _evaluate(self, model, input, target, learningRate):
		if input is not None:
			model.propagateForward( input )
			estimation = model.propagateBackwards( target, learningRate, updateWeights=True )
			return meanSquared( estimation )
		return None

# En el aprendizaje por lotes (batch/off-line), utilizamos el conjunto de datos entero
# para calcular las correcciones, y en este caso el orden de los datos es indistinto
class BatchTraining(TrainingScenario):
	def description(self):
		return 'Batch/Off-line training'

	def executeOn(self, model, input, target, learningRate, stopCondition, validationSet=None, validationTarget=None):
		logger = TrainingLogger()
		logger.logStart(self, model, input, validationSet)

		X = input
		Z = target
		P = input.shape[0]
		epoch = 1
		lastError=1
		self._historicalLoss = [lastError]
		while not stopCondition(lastError, epoch):

			lastError = super()._fit(model, X, Z, learningRate)

			self._historicalLoss.append(lastError)
			epoch += 1

		validationLoss = super()._evaluate(model, validationSet, validationTarget, learningRate)
		logger.logEnd(self, epoch, validationLoss)

	def historicalLoss(self):
		return self._historicalLoss

	def lastLoss(self):
		return self._historicalLoss[-1]


# En el aprendizaje incremental (incremental/on-line) usamos una instancia por vez,
#, y aquí recorrer las instancias en un orden aleatorio suele incrementar
# las chances de converger a una buena solución
class IncrementalTraining(TrainingScenario):
	def description(self):
		return 'Incremental training'

	def executeOn(self, model, input, target, learningRate, stopCondition, validationSet=None, validationTarget=None):
		logger = TrainingLogger()
		logger.logStart(self, model, input, validationSet)

		epoch = 1
		lastError=1
		self._historicalLoss = [lastError]
		while not stopCondition(lastError, epoch):
			lastError=0
			P = input.shape[0]
			for h in range(0,P):
				X_h = num.array([input[h]]) # Solo es un vector. Tiene que ser una matriz para poder transponerse en la correccion
				Z_h = target[h]

				lastError += super()._fit(model, X_h, Z_h, learningRate)

			lastError /= P
			epoch += 1
			self._historicalLoss.append(lastError)

		validationLoss = super()._evaluate(model, validationSet, validationTarget, learningRate)
		logger.logEnd(self, epoch, validationLoss)

	def historicalLoss(self):
		return self._historicalLoss

	def lastLoss(self):
		return self._historicalLoss[-1]

# Compromiso entra ambas técnicas llamada mini-lotes (mini-batch) en donde se eligen al azar porciones
# relativamente chicas de los datos y se los utiliza para calcular las correcciones a los pesos.
class MiniBatchTraining(TrainingScenario):
	def __init__(self, batchSize):
		self._batchSize = batchSize

	def description(self):
		return 'Mini-Batch training with size {}'.format(self._batchSize)

	def executeOn(self, model, input, target, learningRate, stopCondition, validationSet=None, validationTarget=None):
		logger = TrainingLogger()
		logger.logStart(self, model, input, validationSet)

		x = input
		z = target
		P = input.shape[0]
		B = self._batchSize
		epoch = 1
		lastError=1
		self._historicalLoss = [lastError]
		while not stopCondition(lastError, epoch):
			lastError=0
			for j in range( math.floor(P/B) ):	# dividimos el dataset en batches de tamaño B
				H = num.random.permutation(P)
				h = H[0]

				# batch para entrenar
				Xh = x[h:h+B]
				Zh = z[h:h+B]

				lastError += super()._fit(model, Xh, Zh, learningRate)

			self._historicalLoss.append(lastError)
			epoch += 1

		validationLoss = super()._evaluate(model, validationSet, validationTarget, learningRate)
		logger.logEnd(self, epoch, validationLoss)

	def historicalLoss(self):
		return self._historicalLoss

	def lastLoss(self):
		return self._historicalLoss[-1]


class TrainingParser():
	def parse(self, paramsDict):
		if paramsDict['trainingType'] == 'Incremental':
			return IncrementalTraining()
		elif paramsDict['trainingType'] == 'Batch':
			return BatchTraining()
		elif paramsDict['trainingType'] == 'MiniBatch':
			return MiniBatchTraining(batchSize=int(paramsDict['batchSize']))
		else:
			raise Exception('Training type {} not known'.format(paramsDict['trainingType']))
