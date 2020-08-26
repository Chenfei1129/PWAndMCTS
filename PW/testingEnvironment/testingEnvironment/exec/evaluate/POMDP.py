
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

from src.MDPChasing.beliefDepthTransition import SampleNextNode, NodeReward, BeliefReward, SampleNextBelief, TigerReward, TigerTransition, TigerObservation, SampleBeliefReward, BeliefTransition

from src.trajectory import SampleTrajectory, OneStepSampleTrajectory
from src.algorithms.mctsStochasticNew3 import MCTS, ScoreChild,  SelectAction, SelectNextState, InitializeChildren, Expand, ExpandNextState, RollOut, establishPlainActionDist, backup, establishSoftmaxActionDist, establishPlainActionDist
from src.sampleTrajectoryTools.resetObjectsForMultipleTrjaectory import RecordValuesForObjects, ResetObjects, GetObjectsValuesOfAttributes
from src.sampleTrajectoryTools.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
        GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.sampleTrajectoryTools.evaluation import ComputeStatistics 
#rollout add gamma
def main():

    # Node belief 
    rewardParam={'listen_cost':-1, 'open_incorrect_cost':-100, 'open_correct_reward':10}
    rewardFunction=TigerReward(rewardParam)
    
    observationParam={'obs_correct_prob':1, 'obs_incorrect_prob':0}
    observationFunction=TigerObservation(observationParam)
    
    transitionFunction=TigerTransition()
    
    observationSpace=['tiger-left', 'tiger-right', 'Nothing']
    
    sampleNextBelief=SampleNextBelief(transitionFunction, observationFunction, observationSpace)
    
    sampleBeliefReward=SampleBeliefReward(transitionFunction, rewardFunction)
    
    beliefTransition=BeliefTransition(transitionFunction, observationFunction, observationSpace)
    
    beliefReward=BeliefReward(transitionFunction, rewardFunction, observationFunction, observationSpace)
    
    
    node1={'b':{'tiger-left':0.5, 'tiger-right':0.5}, 'depth':1}
    sampleNextNode=SampleNextNode(sampleNextBelief)
    nodeReward=NodeReward(beliefReward)

    actionSpace = ['open-right','listen','open-left']

    isTerminal = lambda state: 0

    cInit = 1
    cBase =100
    scoreChild = ScoreChild(cInit,cBase, nodeReward)
    selectAction = SelectAction(scoreChild)
    selectNextState = SelectNextState(selectAction)


    uniformActionPrior = {action : 1/len(actionSpace) for action in actionSpace}
    getActionPrior = lambda state : uniformActionPrior
    initializeChildren = InitializeChildren(actionSpace, sampleNextNode, getActionPrior)
    expand = Expand( isTerminal, initializeChildren)
    expandNewState = ExpandNextState(sampleNextNode)

    rolloutPolicy = lambda state: random.choice(actionSpace)



    def rolloutHeuristic(beliefState):
        return 0
        
    gamma = 0.5
    maxRolloutStep = 10
    estimateValue = RollOut(rolloutPolicy, maxRolloutStep, sampleNextNode, nodeReward, isTerminal, rolloutHeuristic, gamma)
    numSimulation = 20
    mctsSelectAction = MCTS(numSimulation, selectAction, selectNextState, expand, expandNewState, estimateValue, backup, establishPlainActionDist)

    def sampleAction(state):
        actionDist = mctsSelectAction(state)
        action = maxFromDistribution(actionDist)
        return action

    listen = 0
    openLeft = 0
    openRight = 0

    for i in range(100):
        act = sampleAction(node1)
        if act == 'listen':
            listen = listen +1


        if act == 'open-left':
            openLeft = openLeft +1
        if act == 'open-right':
            openRight = openRight+1
    print(listen, openLeft, openRight)





if __name__ == '__main__':
    main()
