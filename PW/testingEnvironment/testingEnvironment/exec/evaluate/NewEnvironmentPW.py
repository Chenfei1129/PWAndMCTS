import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import random
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import itertools as it
from anytree import AnyNode as Node
import pygame as pg
from pygame.color import THECOLORS

from src.MDPChasing.policies import RandomPolicy
from src.visualization.drawDemo import DrawBackground, DrawCircleOutside, DrawState, ChaseTrialWithTraj, InterpolateState
from src.chooseFromDistribution import SampleFromDistribution, maxFromDistribution
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.MDPChasing.transitionFunction import MultiAgentTransitionInGeneral, MultiAgentTransitionInSwampWorld, MovingAgentTransitionInSwampWorld, StayInBoundaryByReflectVelocity, \
    Reset, IsTerminal, TransitionWithNoise, IsInSwamp

from src.MDPChasing.rewardFunction import RewardFunction
from src.trajectory import SampleTrajectory, OneStepSampleTrajectory
from src.algorithms.pwNew import MCTS, ScoreChild,  SelectAction, SelectNextState, InitializeChildren, Expand, ExpandNextState, PWidening, RollOut, establishPlainActionDist, backup, establishSoftmaxActionDist, establishPlainActionDist


from src.sampleTrajectoryTools.resetObjectsForMultipleTrjaectory import RecordValuesForObjects, ResetObjects, GetObjectsValuesOfAttributes
from src.sampleTrajectoryTools.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
        GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.sampleTrajectoryTools.evaluation import ComputeStatistics

def static (allStates, action): 
    [state, terminalPosition] = allStates
    return terminalPosition
class SampleTrjactoriesForConditions:
    def __init__(self, numTrajectories, saveTrajectoryByParameters):
        self.numTrajectories = numTrajectories
        self.saveTrajectoryByParameters = saveTrajectoryByParameters

    def __call__(self, parameters):
        print(parameters)
        numSimulation = parameters['numSimulation']
        #rolloutHeuristic = parameters['rolloutHeuristic']
        xBoundary = [0, 400]
        yBoundary = [0, 400]
        xSwamp = [175, 225]
        ySwamp = [175, 225]
        swamp = [[xSwamp, ySwamp]]

        noise = parameters['noise']
        cBase = parameters['cBase']
        alpha = parameters['alpha']
        C = parameters['C']
        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
        transitionWithNoise = TransitionWithNoise(noise)

        minDistance = 40
        target = [400, 400]
        isTerminal = IsTerminal(minDistance, target)


        isInSwamp = IsInSwamp(swamp)
    
        singleAgentTransit = MovingAgentTransitionInSwampWorld(transitionWithNoise, stayInBoundaryByReflectVelocity, isTerminal)
    

        transitionFunctionPack = [singleAgentTransit, static]
        multiAgentTransition = MultiAgentTransitionInGeneral(transitionFunctionPack)
        twoAgentTransit = MultiAgentTransitionInSwampWorld(multiAgentTransition, target)


        numOfAgent = 2
        xBoundaryReset = [500, 600]
        yBoundaryReset = [0, 100]
        resetState = Reset([0,400], [ 0,400],numOfAgent, target)

        actionSpace = [(50, 0), (-50, 0), (0, 50), (0, -50)]
    #k = np.random.choice(actionSpace)
    #print(k)

    
        actionCost = -0.02
        swampPenalty = -0.5
        terminalReward = 1
        rewardFunction = RewardFunction(actionCost, terminalReward, swampPenalty, isTerminal, isInSwamp)
    
        maxRunningSteps = 50

        oneStepSampleTrajectory = OneStepSampleTrajectory(twoAgentTransit, rewardFunction)
        sampleTrajecoty = SampleTrajectory(maxRunningSteps, isTerminal, resetState, oneStepSampleTrajectory)
        randomPolicy = RandomPolicy(actionSpace)
        actionDistribution = randomPolicy()
    #numSimulation, selectAction, selectNextState, expand, estimateValue, backup, outputDistribution
       # numSimulation = 50

        cInit = 1
        #cBase =100
        scoreChild = ScoreChild(cInit,cBase)
        selectAction = SelectAction(scoreChild)
        selectNextState = SelectNextState(selectAction)
        uniformActionPrior = {action : 1/4 for action in actionSpace}
        getActionPrior = lambda state : uniformActionPrior
        initializeChildren = InitializeChildren(actionSpace, twoAgentTransit, getActionPrior)
        expand = Expand(isTerminal, initializeChildren)
        #alpha = 0
        #C = 1
        pWidening = PWidening(alpha, C)
        expandNewState = ExpandNextState(twoAgentTransit, pWidening)


        rolloutPolicy = lambda state: random.choice(actionSpace)

        #def rolloutHeuristic(allState):
            #[state, terminalPosition] = allState
            #distanceToTerminal = np.linalg.norm(np.array(target) - np.array(state), ord=2)     
            #return (2*np.exp(-distanceToTerminal/100)-1)


      
        rolloutHeuristic = lambda state: 0#reward return sometimes grab nothing.  

        maxRolloutStep = 15
        estimateValue = RollOut(rolloutPolicy, maxRolloutStep, twoAgentTransit, rewardFunction, isTerminal, rolloutHeuristic)
        mctsSelectAction = MCTS(numSimulation, selectAction, selectNextState, expand, expandNewState, estimateValue, backup, establishPlainActionDist)
    #sampleAction = SampleFromDistribution(actionDictionary)
        def sampleAction(state):
            actionDist = mctsSelectAction(state)
            action = maxFromDistribution(actionDist)
            return action
        trajectoriesWithIntentionDists = []
        for trajectoryId in range(self.numTrajectories):
        	trajectory = sampleTrajecoty(sampleAction)
        	trajectoriesWithIntentionDists.append(trajectory)
        	#print(trajectoriesWithIntentionDists)
        trajectoryFixedParameters = {'Algorithm': "PW"}
        self.saveTrajectoryByParameters(trajectoriesWithIntentionDists, trajectoryFixedParameters, parameters)
        #print(np.mean([len(tra) for tra in trajectoriesWithIntentionDists]))        	
def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numSimulation'] = [ 10, 50, 250]
    manipulatedVariables['noise'] = [  (0.1,0.1)]
    manipulatedVariables['cBase'] = [100] 
    manipulatedVariables['C'] = [1]
    manipulatedVariables['alpha'] = [0]
    


    manipulatedVariables['rolloutHeuristic'] = ['noHeuristic']
   # manipulatedVariables['swamp'] = ['haveSwamp']

     # temp just 2
 # 0 never detect compete, 1 only detect compete
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]


    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'dataNew', 'NewEnvironmentMCTSvsPWLuck3',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajectoryExtension = '.pickle'
    getTrajectorySavePath = lambda trajectoryFixedParameters: GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    saveTrajectoryByParameters = lambda trajectories, trajectoryFixedParameters, parameters: saveToPickle(trajectories, getTrajectorySavePath(trajectoryFixedParameters)(parameters))
   
    numTrajectories = 200
    sampleTrajectoriesForConditions = SampleTrjactoriesForConditions(numTrajectories, saveTrajectoryByParameters)
    [sampleTrajectoriesForConditions(para) for para in parametersAllCondtion]

if __name__ == '__main__':
    main()