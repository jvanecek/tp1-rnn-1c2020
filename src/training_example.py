import math
import numpy as num
from numpy import linalg as LA
#from matplotlib import pyplot as plt, cm
#from matplotlib.ticker import MaxNLocator

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

linearActivation = lambda x: x
signActivation = num.sign
sigmoidActivation = lambda x: 1 / (1 + num.exp(-x))
binaryScale = lambda x: x * 2 - 1
bipolarScale = lambda x: x

readableNames = {
  linearActivation : 'Linear',
  signActivation : 'Sign',
  sigmoidActivation : 'Sigmoid',
  binaryScale : 'Binary',
  bipolarScale : 'Bipolar'
}

def normalized( vector ):
  return (vector-vector.mean(0))/num.square(vector.std(0))

def weightInitializer(numberOfRows,numberOfColumns):
  return num.random.normal( 0, 0.5, (numberOfRows,numberOfColumns) )


def separateDataset(dataset, prctTesting, prctTraining, prctValidation):
  totalOfInstances = dataset.shape[0]
  indices = num.random.permutation( totalOfInstances )
  indTesting = math.floor( totalOfInstances*prctTesting )
  indTraining = math.floor( totalOfInstances*prctTraining )
  indValidation = math.floor( totalOfInstances*prctValidation )

  return [
    dataset[0:indTesting],
    dataset[indTesting:(indTesting+indTraining)],
    dataset[(indTesting+indTraining):]]


def displayDataset( input, expectedOutput ):
  fig = plt.figure()
  xfig = fig.add_subplot(1,2,1, xticklabels=[], yticklabels=[])
  zfig = fig.add_subplot(1,2,2, xticklabels=[], yticklabels=[])

  xfig.matshow( input, cmap=cm.gray)
  zfig.matshow( expectedOutput, cmap=cm.gray)

  plt.show()

def displayLinePlot(points):
  plt.plot(points)
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.show()




logicStatements = num.array( [
  [ 0, 0 ],
  [ 0, 1 ],
  [ 1, 0 ],
  [ 1, 1 ]])

expectedOutput = num.array( [
  [ 0 ],
  [ 1 ],
  [ 1 ],
  [ 0 ]])


P = logicStatements.shape[0]
S = [ logicStatements.shape[1] , 3, expectedOutput.shape[1] ]
L = len(S)

x = logicStatements
z = expectedOutput

# weights
W0 = weightInitializer( S[0]+1, S[1] )
W1 = weightInitializer( S[1]+1, S[2] )

# inicializamos en cero los vectores de activación
Y0 = num.zeros( (1,S[0]+1) )
Y1 = num.zeros( (1,S[1]+1) )
Y2 = num.zeros( (1,S[2]) )

# Training parameters
B = 1 # Instancias para el entrenamiento por mini-lote (1 es igual al incremental)
lr = 0.1
epochs = 30

lastError=1
errorsByEpoch = [lastError]
for i in range( epochs ):
  lastError=0
  for j in range( math.floor(P/B) ):  # dividimos el dataset en batches de tamaño B
    H = num.random.permutation(P)
    h = H[0]

    # batch para entrenar
    Xh = x[h:h+B]
    Zh = z[h:h+B]

    # Forward propagation
    Y0[:] = bias_add( Xh, True )
    Y1[:] = bias_add( num.tanh( num.dot( Y0, W0)), True )
    Y2[:] = num.tanh( num.dot( Y1, W1))

    # Inicializamos las Correccioens
    dW0 = num.zeros_like( W0)
    dW1 = num.zeros_like( W1)

    # Backpropagation
    E2 = Zh-Y2 # Error de la predicción
    dY2 = 1-num.square(Y2) # derivada de Y2 respecto a W1
    D2 = E2*dY2 # correcciones para W1
    dW1 += lr * num.dot( Y1.T, D2)

    E1 = num.dot( D2, W1.T) # Error de la capa 1
    dY1 = 1-num.square(Y1) # derivada de Y1 respecto a W0
    D1 = bias_sub( E1*dY1, True) # correcciones para W0
    dW0 += lr * num.dot( Y0.T, D1)

    # Actualizamos los pesos
    W0 += dW0
    W1 += dW1

    # La expresión general del error es el promedio por lote de la suma de las diferencias cuadradas entre la respuesta deseada y la obtenida para todas las unidades de salida.
    lastError += num.mean( num.sum( num.square( E2 ), axis=1) )

  errorsByEpoch.append(lastError)

print( errorsByEpoch )
# displayLinePlot(errorsByEpoch)