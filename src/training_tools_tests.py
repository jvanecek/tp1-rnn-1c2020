import unittest
from training_tools import WhenReachMinimum, AfterNumberOfTrainings, StopConditionParser


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

class TestStopConditionParser(unittest.TestCase):

	def test_parseAfterNumberOfTrainings(self):
		parser = StopConditionParser()
		parsed = parser.parse( {'stopCondition' : 'AfterNumberOfTrainings', 'maxEpoch' : '20'} )
		self.assertEqual( parsed.__class__.__name__, 'AfterNumberOfTrainings' )
		self.assertEqual( parsed._maxEpoch, 20 )

	def test_parseWhenReachMinimum(self):
		parser = StopConditionParser()
		parsed = parser.parse( {'stopCondition' : 'WhenReachMinimum', 'minimum' : '0.3', 'maxEpoch' : '20' } )
		self.assertEqual( parsed.__class__.__name__, 'WhenReachMinimum' )
		self.assertEqual( parsed._maxEpoch, 20 )
		self.assertEqual( parsed._minimum, 0.3 )

if __name__ == '__main__':
	unittest.main()