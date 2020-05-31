import math
import numpy as num
from numpy import linalg as LA

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
			stopCondition):
		self._subclassResponsibility()

	def historicalLoss(self):
		self._subclassResponsibility()

	def lastLoss(self):
		self._subclassResponsibility()

# En el aprendizaje por lotes (batch/off-line), utilizamos el conjunto de datos entero
# para calcular las correcciones, y en este caso el orden de los datos es indistinto
class BatchTraining(TrainingScenario):
	def description(self):
		return 'Batch/Off-line training'

	def executeOn(self, model, input, target, learningRate, stopCondition):
		X = input
		Z = target
		P = input.shape[0]
		epoch = 1
		lastError=1
		self._historicalLoss = [lastError]
		while not stopCondition(lastError, epoch):
			lastError=0

			model.propagateForward( X )
			E2 = model.propagateBackwards( Z, learningRate )

			lastError += num.mean( num.sum( num.square( E2 ), axis=1) )
			self._historicalLoss.append(lastError)
			epoch += 1

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

	def _norm2sum(self, estimation):
		instancesNorm = [ num.square( LA.norm( instanceError ) ) for instanceError in estimation ]
		return num.sum( instancesNorm )

	def executeOn(self, model, input, target, learningRate, stopCondition):
		epoch = 1
		lastError=1
		self._historicalLoss = [lastError]
		while not stopCondition(lastError, epoch):
			lastError=0
			P = input.shape[0]
			for h in range(0,P):
				X_h = num.array([input[h]]) # Solo es un vector. Tiene que ser una matriz para poder transponerse en la correccion
				Z_h = target[h]

				model.propagateForward( X_h )
				E_h = model.propagateBackwards(Z_h, learningRate)
				lastError += self._norm2sum( E_h )

			lastError /= P
			epoch += 1
			self._historicalLoss.append(lastError)

	def historicalLoss(self):
		return self._historicalLoss

	def lastLoss(self):
		return self._historicalLoss[-1]

# Compromiso entra ambas técnicas llama mini-lotes (mini-batch) en donde se eligen al azar porciones
# relativamente chicas de los datos y se los utiliza para calcular las correcciones a los pesos.
class MiniBatchTraining(TrainingScenario):
	def __init__(self, batchSize):
		self._batchSize = batchSize

	def description(self):
		return 'Mini-Batch training with size {}'.format(self._batchSize)

	def executeOn(self, model, input, target, learningRate, stopCondition):
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

				model.propagateForward( Xh )
				E2 = model.propagateBackwards( Zh, learningRate )

				# La expresión general del error es el promedio por lote de la suma de las
				# diferencias cuadradas entre la respuesta deseada y la obtenida para todas las unidades de salida.
				lastError += num.mean( num.sum( num.square( E2 ), axis=1) )

			self._historicalLoss.append(lastError)
			epoch += 1

	def historicalLoss(self):
		return self._historicalLoss

	def lastLoss(self):
		return self._historicalLoss[-1]
