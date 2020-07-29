

import unittest
import numpy as np
from ddt import ddt, data, unpack
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.visualization.drawValueMap import FindCenterPointState, FindCenterPointValue, DrawValueMap

@ddt
class TestReward(unittest.TestCase):
	def setUp(self):
	    self.background = [[0, 600], [0, 600]]

	@data(
		([1, 1], ([0, 600], [0, 600], [300], [300])),
		([3, 3], ([0, 200, 400, 600], [0, 200, 400, 600], [100, 300, 500], [100, 300, 500])),
		([5, 3], ([0, 120, 240, 360, 480, 600], [0, 200, 400, 600], [60, 180, 300, 420, 540], [100, 300, 500]))
	) 
	@unpack
	def testFindCenterPointState(self, grid, groundTruth):
	    findCenterPointState = FindCenterPointState(self.background)
	    findState = findCenterPointState(grid)
	    self.assertEqual(findState, groundTruth)


	def testFindCenterPointValue(self):
	    findCenterPointState = FindCenterPointState(self.background)
	    def valueFunction(state):
	    	if isInSwamp(state):
	    		return -100
	    	else:
	    		return 10

	    findCenterPointValue = FindCenterPointValue(valueFunction)
	    drawValueMap = DrawValueMap(findCenterPointState, findCenterPointValue, self.background)



if __name__ == '__main__':
    unittest.main()
