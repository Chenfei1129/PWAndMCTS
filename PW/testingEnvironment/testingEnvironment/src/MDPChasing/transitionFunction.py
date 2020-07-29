import numpy as np 
import itertools as it

class Reset():
    def __init__(self, xBoundary, yBoundary, numOfAgent, targetPosition):
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.numOfAgnet = numOfAgent
        self.targetPosition = targetPosition

    def __call__(self):
        xMin, xMax = self.xBoundary
        yMin, yMax = self.yBoundary
        initState = [[np.random.uniform(xMin, xMax),np.random.uniform(yMin, yMax)
                      ], self.targetPosition]
                     
        return initState


class FixedReset():
    def __init__(self, initPositionList):
        self.initPositionList = initPositionList

    def __call__(self, trialIndex):
        initState = self.initPositionList[trialIndex]
        return initState
 

class MultiAgentTransitionInGeneral():
    def __init__(self, allTransitions):
        self.allTransitions = allTransitions

    def __call__(self, allStates, allActions):
        allNewStates = [self.allTransitions[i](allStates, allActions[i]) for i in range(len(self.allTransitions))]
        return allNewStates


class MultiAgentTransitionInSwampWorld():
    def __init__(self, multiAgentTransitionInGeneral, terminalPosition):
        self.multiAgentTransitionInGeneral = multiAgentTransitionInGeneral
        self.terminalPosition = terminalPosition

    def __call__(self, state, action):
        allStates = state
        allActions = [action, [0, 0]]
        allNewStates = self.multiAgentTransitionInGeneral(allStates, allActions)
        return allNewStates


class MovingAgentTransitionInSwampWorld():
    def __init__(self, transitionWithNoise, stayInBoundaryByReflectVelocity, isTerminal):
        self.stayInBoundaryByReflectVelocity = stayInBoundaryByReflectVelocity
        self.transitionWithNoise = transitionWithNoise
        self.isTerminal = isTerminal

    def __call__(self, allStates, action):

        if self.isTerminal(allStates)==True:
            [state, terminalPosition] = allStates
            return state

        else:
            [state, terminalPosition] = allStates
            newState = np.array(state) + np.array(action)
            newStateCheckBoundary, newActionCheckBoundary = self.stayInBoundaryByReflectVelocity(newState, action)
            newaction = newActionCheckBoundary 
            finalNewState = self.transitionWithNoise(newStateCheckBoundary)

            return finalNewState


class TransitionWithNoise():
    def __init__(self, noise):          
        self.noise = noise

    def __call__(self, state):
        x = np.random.normal(state[0], self.noise[0])
        y = np.random.normal(state[1], self.noise[1])
        result = [x, y]
        return result


class StayInBoundaryByReflectVelocity():
    def __init__(self, xBoundary, yBoundary):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary

    def __call__(self, position, velocity):
        adjustedX, adjustedY = position
        adjustedVelX, adjustedVelY = velocity
        if position[0] >= self.xMax:
            adjustedX = 2 * self.xMax - position[0]
            adjustedVelX = -velocity[0]
        if position[0] <= self.xMin:
            adjustedX = 2 * self.xMin - position[0]
            adjustedVelX = -velocity[0]
        if position[1] >= self.yMax:
            adjustedY = 2 * self.yMax - position[1]
            adjustedVelY = -velocity[1]
        if position[1] <= self.yMin:
            adjustedY = 2 * self.yMin - position[1]
            adjustedVelY = -velocity[1]
        checkedPosition = [adjustedX, adjustedY]
        checkedVelocity = [adjustedVelX, adjustedVelY]
        return checkedPosition, checkedVelocity


class CheckBoundary():
    def __init__(self, xBoundary, yBoundary):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary

    def __call__(self, position):
        xPos, yPos = position
        if xPos >= self.xMax or xPos <= self.xMin:
            return False
        elif yPos >= self.yMax or yPos <= self.yMin:
            return False
        return True


class IsInSwamp():
    def __init__(self, swamp):
        self.swamp = swamp

    def __call__(self, state):
        inOrNot = [ (state[0] >= xEachSwamp[0] and state[0] <= xEachSwamp[1] and state[1] >= yEachSwamp[0] and state[1] <= yEachSwamp[1])
             for xEachSwamp, yEachSwamp in self.swamp]
        if True in inOrNot:
            return True
        else:
            return False

        
class IsTerminal():
    def __init__(self, minDistance, terminalPosition):
        self.minDistance = minDistance
        self.terminalPosition = terminalPosition

    def __call__(self, allStates):
        [state, terminalPosition] = allStates
        distanceToTerminal = np.linalg.norm(np.array(self.terminalPosition) - np.array(state), ord=2)     
        return (distanceToTerminal<= self.minDistance)

