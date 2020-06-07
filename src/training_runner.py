import argparse
import logging
import matplotlib.pyplot as plt
from activation import ActivationParser
from configparser import ConfigParser
from perceptron import MultiPerceptronParser
from preprocessing import DatasetParser
from training_scenario import TrainingParser
from training_tools import StopConditionParser

def run_training(configurationFilePath, loglevel):
	conf = ConfigParser()
	conf.read(configurationFilePath)

	logging.basicConfig(
		filename=('{}.log'.format(configurationFilePath[:-4])),
		filemode='w',
		level=getattr(logging, loglevel.upper(), None),
		format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
	logging.getLogger().addHandler(logging.StreamHandler())

	# Dataset Parsing
	trainingSet, validationSet, testingSet, trainingTarget, validationTarget, testingTarget = DatasetParser().parse(conf)

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
