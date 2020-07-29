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
from src.algorithms.mctsStochastic import MCTS, ScoreChild,  SelectAction, SelectNextState, InitializeChildren, Expand, ExpandNextState, RollOut, establishPlainActionDist, backup, establishSoftmaxActionDist, establishPlainActionDist
from src.sampleTrajectoryTools.resetObjectsForMultipleTrjaectory import RecordValuesForObjects, ResetObjects, GetObjectsValuesOfAttributes
from src.sampleTrajectoryTools.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
        GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.sampleTrajectoryTools.evaluation import ComputeStatistics2
def loadFromPickle(path):
    pickleIn = open(path, 'rb')
    object = pickle.load(pickleIn)
    pickleIn.close()
    return object
def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numSimulation'] = [10, 30, 50]
    manipulatedVariables['noise'] = [(10, 10), (30, 30), (50, 50)]
     # temp just 2
 # 0 never detect compete, 1 only detect compete

    trajectoryFixedParameters = {'Algorithm':'MCTS'}
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]
    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateCompeteDetection2',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    def measureSelfReward(trajectories, targetReward=None):
        print(trajectories)
        cumulativeRewardOneTraj = []
        for i in range(numTraj):
            numSteps = len(trajectories[i])
        #print(trajectories[i][0][3])
            cumulativeRewardOneTraj.append([sum(trajectories[i][j][3] for j in range(numSteps))])
        return mean(cumulativeRewardOneTraj)
    measureFunction = lambda df: lambda trajectory: measureSelfReward(trajectory)
    computeStatistics = ComputeStatistics2(loadTrajectoriesFromDf, measureFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    print(statisticsDf) 

if __name__ == '__main__':
    main()
