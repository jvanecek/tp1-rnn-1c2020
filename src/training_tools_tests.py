import unittest
from training_tools import WhenReachMinimum, AfterNumberOfTrainings


class TestWhenReachMinimum(unittest.TestCase):
	def test_evaluateCondition(self):
		shouldStop = WhenReachMinimum(minimum=0.5, maxEpoch=10)

		self.assertTrue( shouldStop(0.5, 3) )
		self.assertFalse( shouldStop(0.6, 3) )
		self.assertTrue( shouldStop(0.6, 10) )
		self.assertTrue( shouldStop(0.6, 11) )

class TestAfterNumberOfTrainings(unittest.TestCase):
	def test_evaluateCondition(self):
		shouldStop = AfterNumberOfTrainings(maxEpoch=10)

		self.assertFalse( shouldStop(0.5, 3) )
		self.assertFalse( shouldStop(0.6, 3) )
		self.assertTrue( shouldStop(0.6, 10) )
		self.assertTrue( shouldStop(0.6, 11) )

if __name__ == '__main__':
	unittest.main()