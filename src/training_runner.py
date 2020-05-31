import argparse
import pandas as pd
import logging
from configparser import ConfigParser
from activation import ActivationParser
from perceptron import MultiPerceptronParser
from training_scenario import TrainingParser
from training_tools import StopConditionParser

def run_training(configurationFilePath, loglevel):
	conf = ConfigParser()
	conf.read(configurationFilePath)

	logging.basicConfig(
		filename=('{}.log'.format(configurationFilePath)),
		filemode='w',
		level=getattr(logging, loglevel.upper(), None),
		format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

	# Dataset Parsing
	header = conf.get('Dataset-Params', 'datasetHeader', fallback=None).split(',')
	trainingSet = conf.get('Dataset-Params', 'trainingSet')
	trainingSet = pd.read_csv( trainingSet, header=None, names=header )

	logging.debug("Raw dataset:\n{}".format(trainingSet))

	targetColumn = conf.get('Dataset-Params', 'targetColumn')
	targetSet = trainingSet.pop( targetColumn )

	logging.debug("Input:\n{}".format( trainingSet ))
	logging.debug("Target:\n{}".format( targetSet ))

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
	parser.add_argument('--log', metavar='string', required=False, default='INFO', help='log level: DEBUG, INFO')
	args = parser.parse_args()

	run_training(configurationFilePath=args.conf, loglevel=args.log)
