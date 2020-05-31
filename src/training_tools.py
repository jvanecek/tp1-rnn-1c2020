
class StopCondition():
	def __call__(self, error, epoch):
		raise Exception('This method should be implemented by subclass')

class WhenReachMinimum(StopCondition):
	def __init__(self, minimum, maxEpoch):
		self._minimum = minimum
		self._maxEpoch = maxEpoch

	def __call__(self, error, epoch):
		return error <= self._minimum or epoch >= self._maxEpoch

class AfterNumberOfTrainings(StopCondition):
	def __init__(self, maxEpoch):
		self._maxEpoch = maxEpoch

	def __call__(self, error, epoch):
		return epoch >= self._maxEpoch


class StopConditionParser():
	def parse(self, paramsDict):
		if paramsDict['stopCondition'] == 'AfterNumberOfTrainings':
			return AfterNumberOfTrainings(maxEpoch=int(paramsDict['maxEpoch']))
		elif paramsDict['stopCondition'] == 'WhenReachMinimum':
			return WhenReachMinimum(
				minimum=float(paramsDict['minimum']),
				maxEpoch=int(paramsDict['maxEpoch']))
		else:
			raise Exception('Stop condition named {} unkwnown'.format(paramsDict['stopCondition']) )