import unittest
import numpy as num
from activation import Tanh
from perceptron import MultiPerceptron
from perceptron_tools import ZeroInitializer
from training_tools import AfterNumberOfTrainings
from training_scenario import BatchTraining, MiniBatchTraining, IncrementalTraining

class TestTrainingBehavior(unittest.TestCase):

	def runXORScenario(self, training):
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

		S = [ logicStatements.shape[1], 3, expectedOutput.shape[1] ]

		model = MultiPerceptron( activation=Tanh(), weightsInitializer=ZeroInitializer() )
		model.configureLayers( S )

		training.executeOn(
			model=model,
			input=logicStatements,
			target=expectedOutput,
			learningRate=0.1,
			stopCondition=AfterNumberOfTrainings(30))

		self.assertEqual( len(training.historicalLoss()), 30 )

class TestBatchTraining(TestTrainingBehavior):
	def test_description(self):
		training = BatchTraining()
		self.assertEqual(training.description(), 'Batch/Off-line training')

	def test_training(self):
		self.runXORScenario( BatchTraining() )

class TestMiniBatchTraining(TestTrainingBehavior):
	def test_description(self):
		training = MiniBatchTraining(batchSize=2)
		self.assertEqual(training.description(), 'Mini-Batch training with size 2')

	def test_trainingWithBatchSizeOne(self):
		self.runXORScenario( MiniBatchTraining(batchSize=1) )

	def test_trainingWithBatchSizeTwo(self):
		self.runXORScenario( MiniBatchTraining(batchSize=2) )

class TestIncrementalTraining(TestTrainingBehavior):
	def test_description(self):
		training = IncrementalTraining()
		self.assertEqual(training.description(), 'Incremental training')

	def test_training(self):
		self.runXORScenario( IncrementalTraining() )

if __name__ == '__main__':
	unittest.main()