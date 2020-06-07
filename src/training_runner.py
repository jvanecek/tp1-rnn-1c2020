import argparse
import logging
import math
import matplotlib.pyplot as plt
import pandas as pd
from activation import ActivationParser
from configparser import ConfigParser
from perceptron import MultiPerceptronParser
from training_scenario import TrainingParser
from training_tools import StopConditionParser

def separateDataset(dataset, prctTraining, prctValidation, prctTesting):
	totalOfInstances = dataset.shape[0]
	#indices = num.random.permutation( totalOfInstances )
	indTraining = math.floor( totalOfInstances*prctTraining )
	indValidation = math.floor( totalOfInstances*prctValidation )
	indTesting = math.floor( totalOfInstances*prctTesting )

	return [
		dataset[0:indTraining],
		dataset[indTraining:(indTraining+indValidation)],
		dataset[(indTraining+indValidation):]]

def parse_datasets(conf):
	header = conf.get('Dataset-Params', 'datasetHeader', fallback=None).split(',')
	rawDataset = conf.get('Dataset-Params', 'trainingSet')
	rawDataset = pd.read_csv( rawDataset, header=None, names=header )
	logging.debug("Raw dataset:\n{}".format(rawDataset))

	datasetPartitions = [float(x) for x in conf.get('Dataset-Params', 'datasetPartitions').split(',')]
	logging.debug("Dataset is gonna be splitted in {}% for Training, {}% for Validations and {}% for Testing".format(datasetPartitions[0]*100,datasetPartitions[1]*100,datasetPartitions[2]*100) )

	targetColumn = conf.get('Dataset-Params', 'targetColumn')
	targetSet = rawDataset.pop( targetColumn )

	logging.debug("Input:\n{}".format( rawDataset ))
	logging.debug("Target:\n{}".format( targetSet ))

	trainingSet, validationSet, testingSet = separateDataset(rawDataset.to_numpy(),datasetPartitions[0],datasetPartitions[1],datasetPartitions[2])
	trainingTarget, validationTarget, testingTarget = separateDataset(targetSet.to_numpy(), datasetPartitions[0],datasetPartitions[1],datasetPartitions[2])
	return trainingSet, validationSet, testingSet, trainingTarget, validationTarget, testingTarget

def run_training(configurationFilePath, loglevel):
	conf = ConfigParser()
	conf.read(configurationFilePath)

	logging.basicConfig(
		filename=('{}.log'.format(configurationFilePath[:-4])),
		filemode='w',
		level=getattr(logging, loglevel.upper(), None),
		format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

	# Dataset Parsing
	trainingSet, validationSet, testingSet, trainingTarget, validationTarget, testingTarget = parse_datasets(conf)

	# Model Parsing
	model = MultiPerceptronParser().parse( conf['Model-Params'], trainingSet.shape[1] )

	# Training Parsing
	learningRate=conf.getfloat('Training-Params', 'learningRate')
	training=TrainingParser().parse( conf['Training-Params'] )
	stopCondition=StopConditionParser().parse( conf['Training-Params'] )

	training.executeOn(
		model=model,
		input=trainingSet,
		target=trainingTarget.reshape(trainingTarget.shape[0],1),
		learningRate=learningRate,
		stopCondition=stopCondition,
		validationSet=validationSet,
		validationTarget=validationTarget.reshape(validationTarget.shape[0],1))

	plt.plot(training.historicalLoss())
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.savefig('{}.png'.format(configurationFilePath[:-4]))
	plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train a neural network')
	parser.add_argument('--conf', metavar='path', required=True, help='the path to the training parameters')
	parser.add_argument('--log', metavar='string', required=False, default='INFO', help='log level: DEBUG, INFO')
	args = parser.parse_args()

	run_training(configurationFilePath=args.conf, loglevel=args.log)
