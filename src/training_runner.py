import argparse
import pandas as pd
from configparser import ConfigParser
from activation import ActivationParser
from perceptron import MultiPerceptronParser
from training_scenario import TrainingParser
from training_tools import StopConditionParser

class TrainingRunner():
	def __init__(self, configurationFilePath):
		conf = ConfigParser()
		conf.read(configurationFilePath)

		# Dataset Parsing
		header = conf.get('Dataset-Params', 'datasetHeader', fallback=None).split(',')
		trainingSet = conf.get('Dataset-Params', 'trainingSet')
		trainingSet = pd.read_csv( trainingSet, header=None, names=header, delimiter=',' )

		targetColumn = conf.get('Dataset-Params', 'targetColumn')
		targetSet = trainingSet.pop( targetColumn )

		# Model Parsing
		model = MultiPerceptronParser().parse( conf['Model-Params'], trainingSet.shape[1] )

		# Training Parsing
		learningRate=conf.getfloat('Training-Params', 'learningRate')
		training=TrainingParser().parse( conf['Training-Params'] )
		stopCondition=StopConditionParser().parse( conf['Training-Params'] )

		training.executeOn(
			model=model,
			input=trainingSet.to_numpy(),
			target=targetSet.to_numpy(),
			learningRate=learningRate,
			stopCondition=stopCondition)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train a neural network')
	parser.add_argument('--conf', metavar='path', required=True, help='the path to the training parameters')
	args = parser.parse_args()
	TrainingRunner(configurationFilePath=args.conf)
