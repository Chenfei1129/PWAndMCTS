import numpy as np
from anytree import AnyNode as Node


class InitializeChildren:
    def __init__(self, actionSpace, transition, getActionPrior):
        self.actionSpace = actionSpace
        self.transition = transition
        self.getActionPrior = getActionPrior

    def __call__(self, node):
        state = list(node.id.values())[0]
        initActionPrior = self.getActionPrior(state)

        for action in self.actionSpace:
            nextState = self.transition(state, action)
            actionNode = Node(parent=node, id={action: action}, numVisited=0, sumValue=0,actionPrior=initActionPrior[action])

        return node

class Expand:
    def __init__(self, isTerminal, initializeChildren):
        self.isTerminal = isTerminal
        self.initializeChildren = initializeChildren

    def __call__(self, leafNode):
        currentState = list(leafNode.id.values())[0]
        if not self.isTerminal(currentState):
            leafNode.isExpanded = True
            leafNode = self.initializeChildren(leafNode)

        return leafNode

class ScoreChild:
    def __init__(self, cInit, cBase, rewardFunction):
        self.cInit = cInit
        self.cBase = cBase
        self.rewardFunction=rewardFunction

    def __call__(self, stateNode, actionNode):
        stateActionVisitCount = actionNode.numVisited
        stateVisitCount = stateNode.numVisited
        actionPrior = actionNode.actionPrior
        if actionNode.numVisited == 0:
            uScore = np.inf
            qScore = 0 
        else:
            explorationRate = np.log((1 + stateVisitCount + self.cBase) / self.cBase) + self.cInit 
            uScore = explorationRate * actionPrior * np.sqrt(stateVisitCount) / float(1 + stateActionVisitCount)#selfVisitCount is stateACtionVisitCount
            nextStateValues = [self.rewardFunction(list(stateNode.id.values())[0], list(actionNode.id.values())[0], list(nextState.id.values())[0])+nextState.sumValue for nextState in actionNode.children]
            qScore = sum(nextStateValues) / stateActionVisitCount
        score = qScore + uScore
        return score
        
class SelectAction:
    def __init__(self, calculateScore):
        self.calculateScore = calculateScore

    def __call__(self, stateNode):
        scores = [self.calculateScore(stateNode, actionNode) for actionNode in list(stateNode.children)]
        maxIndex = np.argwhere(scores == np.max(scores)).flatten()
        selectedChildIndex = np.random.choice(maxIndex)
        selectedAction = stateNode.children[selectedChildIndex]
        return selectedAction

class ExpandNextState:
    def __init__(self, transitionFunction):
        self.transitionFunction = transitionFunction
        
    def __call__(self, stateNode, actionNode):
        state = list(stateNode.id.values())[0]
        action = list(actionNode.id.values())[0]
        nextState = self.transitionFunction(state, action)
        nextStateNode = Node(parent=actionNode, id={action: nextState}, numVisited=0, sumValue=0,
                 isExpanded=False)
        return actionNode.children


class SelectNextState:
    def __init__(self, selectAction):
        self.selectAction = selectAction
        
    def __call__(self, stateNode, actionNode):
        nextPossibleState = actionNode.children
        if actionNode.numVisited == 0:
            probNextStateVisits = [1/len(nextPossibleState) for nextState in nextPossibleState]
            nextState = np.random.choice(nextPossibleState, 1, p =probNextStateVisits)
        else:
            probNextStateVisits = [nextState.numVisited/actionNode.numVisited for nextState in actionNode.children]
            nextState = np.random.choice(nextPossibleState, 1, p =probNextStateVisits)
        return nextState[0]


class RollOut:
    def __init__(self, rolloutPolicy, maxRolloutStep, transitionFunction, rewardFunction, isTerminal, rolloutHeuristic, gamma):
        self.transitionFunction = transitionFunction
        self.rewardFunction = rewardFunction
        self.maxRolloutStep = maxRolloutStep
        self.rolloutPolicy = rolloutPolicy
        self.isTerminal = isTerminal
        self.rolloutHeuristic = rolloutHeuristic
        self.gamma = gamma

    def __call__(self, leafNode):
        currentState = list(leafNode.id.values())[0]
        totalRewardForRollout = 0
        step = 0

        for rolloutStep in range(self.maxRolloutStep):
            action = self.rolloutPolicy(currentState)
            nextState = self.transitionFunction(currentState, action)
            reward = self.rewardFunction(currentState, action, nextState)
            weightedReward = reward*np.power(self.gamma, step)
            totalRewardForRollout += weightedReward
            if self.isTerminal(currentState):
                break

            currentState = nextState
            step = step + 1

        heuristicReward = 0
        if not self.isTerminal(currentState):
            heuristicReward = self.rolloutHeuristic(currentState)
        totalRewardForRollout += heuristicReward

        return totalRewardForRollout

def backup(value, nodeList): #anytree lib
    for node in nodeList:
        node.sumValue += value
        node.numVisited += 1

class MCTS:
    def __init__(self, numSimulation, selectAction, selectNextState, expand, expandNextState, estimateValue, backup, outputDistribution):
        self.numSimulation = numSimulation
        self.selectAction = selectAction 
        self.selectNextState = selectNextState
        self.expand = expand
        self.expandNextState = expandNextState
        self.estimateValue = estimateValue
        self.backup = backup
        self.outputDistribution = outputDistribution

    def __call__(self, currentState):
        root = Node(id={None: currentState}, numVisited=0, sumValue=0, isExpanded=False)
        root = self.expand(root)

        for exploreStep in range(self.numSimulation):
            currentNode = root
            nodePath = [currentNode]

            while currentNode.isExpanded:
                actionNode = self.selectAction(currentNode)
                allNextStateNodes = self.expandNextState(currentNode, actionNode)
                nextStateNode = self.selectNextState(currentNode, actionNode)
                
                nodePath.append(actionNode)
                nodePath.append(nextStateNode)
                currentNode = nextStateNode

            leafNode = self.expand(currentNode)
            value = self.estimateValue(leafNode)
            self.backup(value, nodePath)

        actionDistribution = self.outputDistribution(root)
        return actionDistribution


def establishPlainActionDist(root):
    visits = np.array([child.numVisited for child in root.children])
    actionProbs = visits / np.sum(visits)
    actions = [list(child.id.keys())[0] for child in root.children]
    actionDist = dict(zip(actions, actionProbs))
    return actionDist


def establishSoftmaxActionDist(root):
    visits = np.array([child.numVisited for child in root.children])
    expVisits = np.exp(visits)
    actionProbs = expVisits / np.sum(expVisits)
    actions = [list(child.id.keys())[0] for child in root.children]
    actionDist = dict(zip(actions, actionProbs))
    return actionDist


