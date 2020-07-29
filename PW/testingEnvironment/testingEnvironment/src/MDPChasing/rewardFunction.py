
import numpy as np
		
class RewardFunction():
    def __init__(self, actionCost, terminalReward, swampPenalty, isTerminal, isInSwamp):
        self.actionCost = actionCost
        self.swampPenalty = swampPenalty
        self.terminalReward = terminalReward
        self.isInSwamp = isInSwamp
        self.isTerminal = isTerminal


    def __call__(self, allStates, action, newStates):
        [state, terminalPosition] = allStates
        reward = self.actionCost
        if self.isInSwamp(state)==True: 
            reward+=self.swampPenalty
        if self.isTerminal(allStates)==True:
            reward += self.terminalReward
        return reward
