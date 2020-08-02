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

           # Node(parent=actionNode, id={action: nextState}, numVisited=0, sumValue=0,
                 #isExpanded=False)

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
    def __init__(self, cInit, cBase):
        self.cInit = cInit
        self.cBase = cBase
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
            nextStateValues = [nextState.sumValue for nextState in actionNode.children]
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
        #print(nextPossibleState)
        if actionNode.numVisited == 0:
            probNextStateVisits = [1/len(nextPossibleState) for nextState in nextPossibleState]
            #print(nextPossibleState)
            #print(probNextStateVisits)
            nextState = np.random.choice(nextPossibleState, 1, p =probNextStateVisits)
        else:
            probNextStateVisits = [nextState.numVisited/actionNode.numVisited for nextState in actionNode.children]
           # print("A")
            #print(nextPossibleState)
            nextState = np.random.choice(nextPossibleState, 1, p =probNextStateVisits)
            #print(nextState)
            #print("B")
        return nextState[0]


class RollOut:
    def __init__(self, rolloutPolicy, maxRolloutStep, transitionFunction, rewardFunction, isTerminal, rolloutHeuristic):
        self.transitionFunction = transitionFunction
        self.rewardFunction = rewardFunction
        self.maxRolloutStep = maxRolloutStep
        self.rolloutPolicy = rolloutPolicy
        self.isTerminal = isTerminal
        self.rolloutHeuristic = rolloutHeuristic

    def __call__(self, leafNode):
        currentState = list(leafNode.id.values())[0]
        totalRewardForRollout = 0

        for rolloutStep in range(self.maxRolloutStep):
            action = self.rolloutPolicy(currentState)
            nextState = self.transitionFunction(currentState, action)
            totalRewardForRollout += self.rewardFunction(currentState, action, nextState)
            if self.isTerminal(currentState):
                #print("***")
                break

            currentState = nextState

        heuristicReward = 0
        if not self.isTerminal(currentState):
            heuristicReward = self.rolloutHeuristic(currentState)
            #print(heuristicReward)
        totalRewardForRollout += heuristicReward

        return totalRewardForRollout

def backup(value, nodeList): #anytree lib
    for node in nodeList:
        node.sumValue += value
        node.numVisited += 1

class MCTS:
    def __init__(self, numSimulation, selectAction, selectNextState, expand, expandNewState, estimateValue, backup, outputDistribution):
        self.numSimulation = numSimulation
        self.selectAction = selectAction 
        self.selectNextState = selectNextState
        self.expand = expand
        self.expandNewState = expandNewState
        self.estimateValue = estimateValue
        self.backup = backup
        self.outputDistribution = outputDistribution

    def __call__(self, currentState):
        #print(currentState)
        root = Node(id={None: currentState}, numVisited=0, sumValue=0, isExpanded=False)
        root = self.expand(root)

        for exploreStep in range(self.numSimulation):
            currentNode = root
            nodePath = [currentNode]

            while currentNode.isExpanded:
                actionNode = self.selectAction(currentNode)
                #print(actionNode)
                allNextStateNodes = self.expandNewState(currentNode, actionNode)
                #print(allNextStateNodes)
                nextStateNode = self.selectNextState(currentNode, actionNode)
                #print(nextStateNode)
                
                nodePath.append(actionNode)
                nodePath.append(nextStateNode)
                currentNode = nextStateNode

            leafNode = self.expand(currentNode)
            value = self.estimateValue(leafNode)
            self.backup(value, nodePath)

        actionDistribution = self.outputDistribution(root)
        #print(actionDistribution)
        #print(actionDistribution)
        return actionDistribution

class RewardFunction:
    def __init__(self, step_penalty, catch_reward, isTerminal):
        self.step_penalty = step_penalty
        self.catch_reward = catch_reward
        self.isTerminal = isTerminal

    def __call__(self, state, action):
        if self.isTerminal(state):
            return self.catch_reward
        else:
            return self.step_penalty

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
'''
currentState = [[0,1],[0,1]]
root = Node(id={"state": currentState}, numVisited=0, sumValue=0, isExpanded=True)


currentNode = root
nodePath = [currentNode]
if currentNode.isExpanded:
    print("*")
'''

