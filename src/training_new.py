import math
import numpy as num
from numpy import linalg as LA
from perceptron import MultiPerceptron
from perceptron_tools import RandomInitializer, ZeroInitializer
from activation import Tanh
from training_scenario import MiniBatchTraining
from training_tools import AfterNumberOfTrainings

weightInitializer = ZeroInitializer()

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
S = [ logicStatements.shape[1], 3, 2, 2, expectedOutput.shape[1] ]

x = logicStatements
z = expectedOutput

model = MultiPerceptron( activation=Tanh(), weightsInitializer=weightInitializer )
model.configureLayers( S )

training = MiniBatchTraining(batchSize=1)
training.executeOn(
	model=model,
	input=x,
	target=z,
	learningRate=0.1,
	stopCondition=AfterNumberOfTrainings(30))

print( training.historicalLoss() )
# displayLinePlot(errorsByEpoch)