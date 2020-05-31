
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
