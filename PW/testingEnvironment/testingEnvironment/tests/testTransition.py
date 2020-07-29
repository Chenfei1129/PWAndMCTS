
import unittest
import numpy as np
from ddt import ddt, data, unpack
import sys 
import random
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Local import
from src.MDPChasing.transitionFunction import  StayInBoundaryByReflectVelocity, CheckBoundary, TransitionWithNoise, IsInSwamp, IsTerminal, MovingAgentTransitionInSwampWorld
@ddt
class TestEnvNoPhysics(unittest.TestCase):
    def setUp(self):
        self.numOfAgent = 2
        self.minDistance = 50
        self.terminalPosition = [50, 50]
        self.xBoundary = [0, 640]
        self.yBoundary = [0, 480]
        self.isTerminalSingleAgent = IsTerminal(self.minDistance, self.terminalPosition)
        self.stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(
            self.xBoundary, self.yBoundary)

    @data(([0, 0], [0, 0], [0, 0], [0, 0]), 
          ([1, 1], [0, 0], [0, 0], [1, 1]),
          ([1, 1], [1, 2], [1, 2], [1, 1]))         
    @unpack
    def testTransitionWithNoiseMean(self, standardDeviation, mu, groundTruthSampleMean, groundTruthsampleStandardDeviation):
        transitionWithNoise = TransitionWithNoise(standardDeviation)
        nextStates = [transitionWithNoise(mu) for _ in range(1000)]
        samplemean = [sum(nextstate[0] for nextstate in nextStates)/len(nextStates), sum(nextstate[1] for nextstate in nextStates)/len(nextStates)]
        standardDeviationResult = [np.std(nextStates[0]), np.std(nextStates[1])]
        truthValue = abs(np.array(samplemean) - np.array(groundTruthSampleMean))< 0.1
        self.assertTrue(truthValue.all())

    @data(([0, 0], [0, 0], [0, 0], [0, 0]), 
          ([2, 2], [0, 0], [0, 0], [2, 2]),
          ([1, 1], [1, 2], [1, 2], [1, 1]))         
    @unpack
    def testTransitionWithNoiseStandard(self, standardDeviation, mu, groundTruthSampleMean, groundTruthsampleStandardDeviation):
        transitionWithNoise = TransitionWithNoise(standardDeviation)
        nextStates = [transitionWithNoise(mu) for _ in range(1000)]
        samplemean = [sum(nextstate[0] for nextstate in nextStates)/len(nextStates), sum(nextstate[1] for nextstate in nextStates)/len(nextStates)]
        xNextState = [nextstate[0] for nextstate in nextStates]
        yNextState = [nextstate[1] for nextstate in nextStates]
        standardDeviationResult = [np.std(xNextState), np.std(yNextState)]       
        truthValue = abs(np.array(standardDeviationResult) - np.array(groundTruthsampleStandardDeviation))< 0.1
        self.assertTrue(truthValue.all())

    @data(([0, 0], [0, 0], [0, 0]), ([1, -2], [1, -3], [1, 2]), ([1, 3], [2, 2], [1, 3]))
    @unpack
    def testCheckBoundaryAndAdjust(self, state, action, groundTruthNextState):
        checkState, checkAction = self.stayInBoundaryByReflectVelocity(state, action)
        truthValue = checkState == groundTruthNextState
        self.assertTrue(truthValue)
 
    @data(([1, 1], True), ([1, -2], False), ([650, 120], False))
    @unpack
    def testCheckBoundary(self, position, groundTruth):
        self.checkBoundary = CheckBoundary(self.xBoundary, self.yBoundary)
        returnedValue = self.checkBoundary(position)
        truthValue = returnedValue == groundTruth
        self.assertTrue(truthValue)
    
    @data(([0, 0], False),([50, 50], False),([450, 10],  True))
    @unpack
    def testInSwamp(self, state, expectedResult):
        swamp = [[[100, 200], [150, 250]],[[400, 450], [0, 10]]]
        isInSwamp = IsInSwamp(swamp)
        checkInSwamp = isInSwamp(state)
        truthValue = checkInSwamp == expectedResult
        self.assertTrue(truthValue)

    @data(([[0, 50], [50, 50]], True), ([[25, 25], [50, 50]],  True), ([[100, 2],[50, 50]], False), ( [[300, 300], [50, 50]], False))
    @unpack
    def testTerminal(self, state,  groundTruth):
        inTerminal = self.isTerminalSingleAgent(state)
        truthValue = inTerminal == groundTruth
        self.assertTrue(truthValue)


    @data(([0, 0], [[0, 0], [200, 200]], [0, 0], [0, 0]), 
          ([1, 1], [[1, 1], [200, 200]], [1, 2], [2, 3]),
          ([0, 0], [[640, 2], [30, 30]], [1, 0], [639, 2]),
          ([0, 0], [[640, 2], [30, 30]], [1, -3], [639, 1]))
    @unpack

    def testSingleAgentTransition(self, standardDeviation, state, action, groundTruthReturnedNextStateMean):   
        transitionWithNoise = TransitionWithNoise (standardDeviation)
        transition = MovingAgentTransitionInSwampWorld(transitionWithNoise, self.stayInBoundaryByReflectVelocity, self.isTerminalSingleAgent)
        nextStates = [transition(state, action) for _ in range(1000)]

        sampleMean = [sum(nextstate[0] for nextstate in nextStates)/len(nextStates), sum(nextstate[1] for nextstate in nextStates)/len(nextStates)] 
        truthValue = abs(np.array(sampleMean) - np.array(groundTruthReturnedNextStateMean))<0.1
        self.assertTrue(truthValue.all()) 



if __name__ == '__main__':
    unittest.main()
