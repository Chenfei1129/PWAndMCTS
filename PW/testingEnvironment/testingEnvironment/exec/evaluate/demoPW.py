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
from src.algorithms.pw import MCTS, PWidening, ScoreChild,  SelectAction, SelectNextState, InitializeChildren, Expand, RollOut, establishPlainActionDist, backup, establishSoftmaxActionDist, establishPlainActionDist

def static (allStates, action): 
    [state, terminalPosition] = allStates
    return terminalPosition

def main():

    # MDP Env
    xBoundary = [0, 600]
    yBoundary = [0, 600]
    xSwamp = [300, 400]
    ySwamp = [300, 400]
    swamp = [[[300,400],[300,400]]]

    noise = [1, 1]
    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    transitionWithNoise = TransitionWithNoise(noise)

    minDistance = 50
    target = [500, 500]
    isTerminal = IsTerminal(minDistance, target)


    isInSwamp = IsInSwamp(swamp)
    
    singleAgentTransit = MovingAgentTransitionInSwampWorld(transitionWithNoise, stayInBoundaryByReflectVelocity, isTerminal)
    

    transitionFunctionPack = [singleAgentTransit, static]
    multiAgentTransition = MultiAgentTransitionInGeneral(transitionFunctionPack)
    twoAgentTransit = MultiAgentTransitionInSwampWorld(multiAgentTransition, target)


    numOfAgent = 2
    xBoundaryReset = [500, 600]
    yBoundaryReset = [0, 100]
    resetState = Reset([0,0], [0,0], numOfAgent, target)

    actionSpace = [(100, 0), (-100, 0), (0, 100), (0, -100)]
    #k = np.random.choice(actionSpace)
    #print(k)

    
    actionCost = -1
    swampPenalty = -100
    terminalReward = 100
    rewardFunction = RewardFunction(actionCost, terminalReward, swampPenalty, isTerminal, isInSwamp)
    
    maxRunningSteps = 50

    oneStepSampleTrajectory = OneStepSampleTrajectory(twoAgentTransit, rewardFunction)
    sampleTrajecoty = SampleTrajectory(maxRunningSteps, isTerminal, resetState, oneStepSampleTrajectory)
    randomPolicy = RandomPolicy(actionSpace)
    actionDistribution = randomPolicy()
    #numSimulation, selectAction, selectNextState, expand, estimateValue, backup, outputDistribution
    numSimulation = 20
    cInit = 100
    cBase =1
    scoreChild = ScoreChild(cInit,cBase)
    selectAction = SelectAction(scoreChild)
    selectNextState = SelectNextState(selectAction)
    uniformActionPrior = {action : 1/4 for action in actionSpace}
    getActionPrior = lambda state : uniformActionPrior

    alpha = 0
    C = 0
    pw = PWidening(alpha, C)

    initializeChildren = InitializeChildren(actionSpace, twoAgentTransit, getActionPrior)
    expand = Expand(isTerminal, initializeChildren)
    rolloutPolicy = lambda state: random.choice(actionSpace)
    rolloutHeuristic = lambda state: 0#reward return sometimes grab nothing.  
    maxRolloutStep = 50
    estimateValue = RollOut(rolloutPolicy, maxRolloutStep, twoAgentTransit, rewardFunction, isTerminal, rolloutHeuristic)
    mctsSelectAction = MCTS(numSimulation, selectAction, selectNextState, expand, estimateValue, backup, establishPlainActionDist)
    #sampleAction = SampleFromDistribution(actionDictionary)
    def sampleAction(state):
        actionDist = mctsSelectAction(state)
        action = maxFromDistribution(actionDist)
        return action

    

    trajectories = [sampleTrajecoty(sampleAction) for _ in range(1)]
    print(trajectories)

    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateObstacle2',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    

    # generate demo image
    screenWidth = 600
    screenHeight = 600
    screen = pg.display.set_mode((screenWidth, screenHeight))
    screenColor = THECOLORS['black']
    xBoundary = [0, 600]
    yBoundary = [0, 600]
    lineColor = THECOLORS['white']
    lineWidth = 4
    xSwamp=[300,400]
    ySwamp=[300,400]
    drawBackground = DrawBackground(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth, xSwamp, ySwamp)

    fps=40
    circleColorSpace = np.array([[0, 0, 255], [0, 255, 255] ])
    circleSize = 10
    positionIndex = [0, 1]
    
    saveImage = True
    imageSavePath = os.path.join(trajectoryDirectory, 'picMovingSheep')
    if not os.path.exists(imageSavePath):
        os.makedirs(imageSavePath)
    trajectoryParameters = 'obstacle'
    imageFolderName = str(trajectoryParameters)

    saveImageDir = os.path.join(os.path.join(imageSavePath, imageFolderName))
    if not os.path.exists(saveImageDir):
        os.makedirs(saveImageDir)
    agentIdsToDraw = list(range(2))
    drawState = DrawState(fps, screen, circleColorSpace, circleSize, agentIdsToDraw, positionIndex, 
            saveImage, saveImageDir, drawBackground)
    
    numFramesToInterpolate = 3
    interpolateState = InterpolateState(numFramesToInterpolate, twoAgentTransit)

    stateIndexInTimeStep = 0
    actionIndexInTimeStep = 1
    
    chaseTrial = ChaseTrialWithTraj(stateIndexInTimeStep, drawState, interpolateState, actionIndexInTimeStep)
   
    [chaseTrial(trajectory) for trajectory in trajectories]
    pg.quit()

    
if __name__ == '__main__':
    main()