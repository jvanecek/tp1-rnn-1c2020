import unittest
import numpy as num
from activation import Tanh, Sigmoid
from perceptron import MultiPerceptron
from perceptron_tools import ZeroInitializer
from training_tools import AfterNumberOfTrainings, WhenReachMinimum
from training_scenario import BatchTraining, MiniBatchTraining, IncrementalTraining, TrainingParser

class TestTrainingBehavior(unittest.TestCase):

	def trainXORWithTanh(self, training, stopCondition):
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
			stopCondition=stopCondition)

		return model

	def trainXORWithSigmoid(self, training, stopCondition):
		logicStatements = num.array( [
			[ 0, 0, 0 ],
			[ 0, 1, 1 ],
			[ 1, 0, 0 ],
			[ 1, 1, 1 ],
			[ 1, 0, 1 ]])
		expectedOutput = num.array( [
			[ 0 ],
			[ 0 ],
			[ 1 ],
			[ 1 ],
			[ 0 ]])

		S = [ logicStatements.shape[1], 3, 2, expectedOutput.shape[1] ]

		model = MultiPerceptron( activation=Sigmoid(), weightsInitializer=ZeroInitializer() )
		model.configureLayers( S )

		training.executeOn(
			model=model,
			input=logicStatements,
			target=expectedOutput,
			learningRate=0.1,
			stopCondition=stopCondition)

		return model

	def trainAndEvaluateAgainstValidationSet(self, training, stopCondition):
		logicStatements = num.array( [
			[ 0, 0, 0 ],
			[ 0, 1, 1 ],
			[ 1, 0, 0 ]])
		expectedOutput = num.array( [
			[ 0 ],
			[ 0 ],
			[ 1 ]])

		validationSet = num.array( [
			[ 1, 1, 1 ],
			[ 1, 0, 1 ]])
		validationTarget = num.array( [
			[ 1 ],
			[ 0 ]])

		S = [ logicStatements.shape[1], 3, 2, expectedOutput.shape[1] ]

		model = MultiPerceptron( activation=Sigmoid(), weightsInitializer=ZeroInitializer() )
		model.configureLayers( S )

		training.executeOn(
			model=model,
			input=logicStatements,
			target=expectedOutput,
			learningRate=0.1,
			stopCondition=stopCondition,
			validationSet=validationSet,
			validationTarget=validationTarget)

		return model

class TestBatchTraining(TestTrainingBehavior):
	def test_description(self):
		training = BatchTraining()
		self.assertEqual(training.description(), 'Batch/Off-line training')

	def test_training_tanh(self):
		training = BatchTraining()
		self.trainXORWithTanh( training, AfterNumberOfTrainings(30) )
		self.assertEqual( len(training.historicalLoss()), 30)

	def test_training_sigmoid(self):
		training = BatchTraining()
		self.trainXORWithSigmoid( training, AfterNumberOfTrainings(35) )
		self.assertEqual( len(training.historicalLoss()), 35)
		self.assertTrue( training.lastLoss() < 0.25 )

	def test_training_tanh_reachminimum(self):
		training = BatchTraining()
		self.trainXORWithTanh( training, WhenReachMinimum(minimum=0.3, maxEpoch=100) )
		self.assertEqual( len(training.historicalLoss()), 4)
		self.assertTrue( training.lastLoss() < 0.3 )

	def test_training_sigmoid_whenreachminimum(self):
		training = BatchTraining()
		self.trainXORWithSigmoid( training, WhenReachMinimum(minimum=0.3, maxEpoch=100) )
		self.assertEqual( len(training.historicalLoss()), 2)
		self.assertTrue( training.lastLoss() < 0.3 )

	def test_training_and_evaluate(self):
		training = BatchTraining()
		self.trainAndEvaluateAgainstValidationSet( training, AfterNumberOfTrainings(35) )
		self.assertEqual( len(training.historicalLoss()), 35)
		self.assertTrue( training.lastLoss() < 0.25 )

class TestMiniBatchTraining(TestTrainingBehavior):
	def test_description(self):
		training = MiniBatchTraining(batchSize=2)
		self.assertEqual(training.description(), 'Mini-Batch training with size 2')

		training = MiniBatchTraining(batchSize=3)
		self.assertEqual(training.description(), 'Mini-Batch training with size 3')

	def test_training_batchsizeOne_tanh(self):
		training = MiniBatchTraining(batchSize=1)
		self.trainXORWithTanh( training, AfterNumberOfTrainings(30) )
		self.assertEqual( len(training.historicalLoss()), 30)

	def test_training_batchsizeOne_sigmoid(self):
		training = MiniBatchTraining(batchSize=1)
		self.trainXORWithSigmoid( training, AfterNumberOfTrainings(20) )
		self.assertEqual( len(training.historicalLoss()), 20)

	def test_training_batchsizeTwo_tanh(self):
		training = MiniBatchTraining(batchSize=2)
		self.trainXORWithTanh( training, AfterNumberOfTrainings(40) )
		self.assertEqual( len(training.historicalLoss()), 40 )

	def test_training_batchsizeTwo_sigmoid(self):
		training = MiniBatchTraining(batchSize=2)
		self.trainXORWithSigmoid( training, AfterNumberOfTrainings(30) )
		self.assertEqual( len(training.historicalLoss()), 30)

	def test_training_batchsizeTwo_tanh_whenreachminimum(self):
		training = MiniBatchTraining(batchSize=2)
		self.trainXORWithTanh( training, WhenReachMinimum(minimum=0.3, maxEpoch=100) )
		# No es deterministico porque mini batch tiene un componente random
		self.assertTrue(\
			len(training.historicalLoss()) == 100 or\
			training.lastLoss() < 0.3 )

	def test_training_batchsizeTwo_sigmoid_whenreachminimum(self):
		training = MiniBatchTraining(batchSize=2)
		self.trainXORWithSigmoid( training, WhenReachMinimum(minimum=0.3, maxEpoch=100) )
		# No es deterministico porque mini batch tiene un componente random
		self.assertTrue(\
			len(training.historicalLoss()) == 100 or\
			training.lastLoss() < 0.3 )

	def test_training_and_evaluate(self):
		training = MiniBatchTraining(batchSize=2)
		self.trainAndEvaluateAgainstValidationSet( training, AfterNumberOfTrainings(35) )
		self.assertEqual( len(training.historicalLoss()), 35)

class TestIncrementalTraining(TestTrainingBehavior):
	def test_description(self):
		training = IncrementalTraining()
		self.assertEqual(training.description(), 'Incremental training')

	def test_training_tanh(self):
		training = IncrementalTraining()
		self.trainXORWithTanh( training, AfterNumberOfTrainings(30) )
		self.assertEqual( len(training.historicalLoss()), 30)

	def test_training_sigmoid(self):
		training = IncrementalTraining()
		self.trainXORWithSigmoid( training, AfterNumberOfTrainings(12) )
		self.assertEqual( len(training.historicalLoss()), 12)

	def test_training_tanh_whenreachminimum(self):
		training = IncrementalTraining()
		self.trainXORWithTanh( training, WhenReachMinimum(minimum=0.3, maxEpoch=100) )
		# No es deterministico porque incremental tiene un componente random
		self.assertTrue(\
			len(training.historicalLoss()) == 100 or\
			training.lastLoss() < 0.3 )

	def test_training_sigmoid_whenreachminimum(self):
		training = IncrementalTraining()
		self.trainXORWithSigmoid( training, WhenReachMinimum(minimum=0.3, maxEpoch=100) )
		# No es deterministico porque incremental tiene un componente random
		self.assertTrue(\
			len(training.historicalLoss()) == 100 or\
			training.lastLoss() < 0.3 )

	def test_training_and_evaluate(self):
		training = IncrementalTraining()
		self.trainAndEvaluateAgainstValidationSet( training, AfterNumberOfTrainings(35) )
		self.assertEqual( len(training.historicalLoss()), 35)

class TestTrainingParser(unittest.TestCase):
	def test_parseIncremental(self):
		params = {'trainingType' : 'Incremental'}
		training = TrainingParser().parse(params)
		self.assertEqual( training.description(), 'Incremental training' )

	def test_parseBatch(self):
		params = {'trainingType' : 'Batch'}
		training = TrainingParser().parse(params)
		self.assertEqual( training.description(), 'Batch/Off-line training' )

	def test_parseMiniBatch(self):
		params = {'trainingType' : 'MiniBatch', 'batchSize' : '30'}
		training = TrainingParser().parse(params)
		self.assertEqual( training.description(), 'Mini-Batch training with size 30' )

if __name__ == '__main__':
	unittest.main()
