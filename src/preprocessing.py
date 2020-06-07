import math
import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def bipolar_scaling( dataset ):
	scaler = MinMaxScaler()
	return pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)

def binary_scaling( dataset ):
	scaledDataset = bipolar_scaling( dataset )
	return scaledDataset * 2 - 1

def parse_scaling( conf ):
	scalingName = conf.get('Dataset-Params', 'scaling')
	if scalingName == 'Binary':
		return binary_scaling
	if scalingName == 'Bipolar':
		return bipolar_scaling
	raise Exception('Not scaling known as {}'.format(scalingName))

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

class DatasetParser():
	def parse(self, conf):
		header = conf.get('Dataset-Params', 'datasetHeader', fallback=None).split(',')
		rawDataset = conf.get('Dataset-Params', 'trainingSet')
		rawDataset = pd.read_csv( rawDataset, header=None, names=header )
		logging.debug("Raw dataset:\n{}".format(rawDataset))

		scaledDataset = parse_scaling(conf)(rawDataset)
		logging.debug("Scaled dataset:\n{}".format(scaledDataset))

		datasetPartitions = [float(x) for x in conf.get('Dataset-Params', 'datasetPartitions').split(',')]
		logging.debug("Dataset is gonna be splitted in {}% for Training, {}% for Validations and {}% for Testing".format(datasetPartitions[0]*100,datasetPartitions[1]*100,datasetPartitions[2]*100) )

		targetColumn = conf.get('Dataset-Params', 'targetColumn')
		targetSet = scaledDataset.pop( targetColumn )

		logging.debug("Input:\n{}".format( scaledDataset ))
		logging.debug("Target:\n{}".format( targetSet ))

		trainingSet, validationSet, testingSet = separateDataset(scaledDataset.to_numpy(),datasetPartitions[0],datasetPartitions[1],datasetPartitions[2])
		trainingTarget, validationTarget, testingTarget = separateDataset(targetSet.to_numpy(), datasetPartitions[0],datasetPartitions[1],datasetPartitions[2])
		return trainingSet, validationSet, testingSet, trainingTarget, validationTarget, testingTarget
