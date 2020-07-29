
import numpy as np
import random

# one step input: reward, transition(init) current state(call) output:list/dict state action next state reward
class OneStepSampleTrajectory:
    def __init__(self, transitionFunction, rewardFunction):
        self.transitionFunction = transitionFunction
        self.rewardFunction = rewardFunction


    def __call__(self, state, sampleAction):
        action = sampleAction(state)
        #print(action)
        nextState = self.transitionFunction(state, action)
        reward = self.rewardFunction(state, action, nextState)
        return (state, action, nextState, reward)


class SampleTrajectory:
    def __init__(self, maxRunningSteps, isTerminal, resetState, forwardOneStep):
        self.maxRunningSteps = maxRunningSteps
        self.isTerminal = isTerminal
        self.resetState = resetState
        self.forwardOneStep = forwardOneStep


    def __call__(self, sampleAction):# update sampleAction.             
        state = self.resetState()
        while self.isTerminal(state):
            state = self.resetState()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None, 1000))
                break
            state, action, nextState, reward = self.forwardOneStep(state, sampleAction)
            trajectory.append((state, action, nextState, reward))
            state = nextState


        return trajectory

class SampleTrajectory2:
    def __init__(self, maxRunningSteps, isTerminal, resetState, forwardOneStep):
        self.maxRunningSteps = maxRunningSteps
        self.isTerminal = isTerminal
        self.resetState = resetState
        self.forwardOneStep = forwardOneStep


    def __call__(self, sampleAction):# update sampleAction.             
        state = self.resetState
        while self.isTerminal(state):
            state = self.resetState()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None, 1000))
                break
            state, action, nextState, reward = self.forwardOneStep(state, sampleAction)
            trajectory.append((state, action, nextState, reward))
            state = nextState


        return trajectory
class GetState:
    def __init__(self, maxRunningSteps, isTerminal, resetState, forwardOneStep):
        self.maxRunningSteps = maxRunningSteps
        self.isTerminal = isTerminal
        self.resetState = resetState
        self.forwardOneStep = forwardOneStep


    def __call__(self, sampleAction):# update sampleAction.             
        state = self.resetState()
        while self.isTerminal(state):
            state = self.resetState()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state))
                break
            state, action, nextState, reward = self.forwardOneStep(state, sampleAction)
            trajectory.append((state,nextState))
            state = nextState


        return trajectory
