

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.visualization.drawValueMap import FindCenterPointState, FindCenterPointValue, DrawValueMap
from src.MDPChasing.transitionFunction import IsInSwamp, IsTerminal

background = [[0, 600], [0, 600]]
findCenterPointState = FindCenterPointState(background)
TerminalPosition = [200, 200]
swamp = [[[300, 400], [300, 400]]]
minDistance = 50
isTerminal = IsTerminal(minDistance, TerminalPosition)
isInSwamp = IsInSwamp(swamp)
def valueFunction(state):
	if isInSwamp(state):
	    return -100
	if isTerminal([state, TerminalPosition]):
		return 10
	else:
	    return 0
findCenterPointValue = FindCenterPointValue(valueFunction)
drawValueMap = DrawValueMap(findCenterPointState, findCenterPointValue, background)
drawValueMap([10, 10])