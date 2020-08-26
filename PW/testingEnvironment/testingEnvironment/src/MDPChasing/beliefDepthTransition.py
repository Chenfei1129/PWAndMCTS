# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:07:56 2020

@author: Kevin
"""

import numpy as np

class SampleNextBelief(object):
    
    def __init__(self, transitionFunction, observationFunction, observationSpace):
        self.transitionFunction=transitionFunction
        self.observationFunction=observationFunction
        self.observationSpace=observationSpace
        
    def __call__(self, b, a):
        s=np.random.choice(list(b.keys()), p=list(b.values()))
        sPrimeProb={sPrime: self.transitionFunction(s, a, sPrime) for sPrime in b.keys()}
        sPrime=np.random.choice(list(sPrimeProb.keys()), p=list(sPrimeProb.values()))
        oProb={o: self.observationFunction(sPrime, a, o) for o in self.observationSpace}
        o=np.random.choice(list(oProb.keys()), p=list(oProb.values()))
        bPrime=self.se(b,a,o)
        return bPrime
    
    def expect(self, xDistribution, function):
        expectation=sum([function(x)*px for x, px in xDistribution.items()])
        return expectation
          
    def se(self, b,a,o):
        bPrimeUnormalized={sPrime: self.observationFunction(sPrime, a, o)*self.expect(b, lambda s: self.transitionFunction(s, a, sPrime)) for sPrime in b}
        alpha=sum(bPrimeUnormalized.values())
        if alpha==0:
            return np.NaN
        bPrime={sPrime: bSPrimeUnormalized/alpha for sPrime, bSPrimeUnormalized in bPrimeUnormalized.items()}
        return bPrime


class BeliefTransition(object):
    
    def __init__(self, transitionFunction, observationFunction, observationSpace):
        self.transitionFunction=transitionFunction
        self.observationFunction=observationFunction
        self.observationSpace=observationSpace
        
    def __call__(self, b, a, bPrime):
        prob=0
        stateSpace=list(b.keys())
        for o in self.observationSpace:
            probByO=0
            for sPrime in stateSpace:
                probBySPrime=0
                for s in stateSpace:
                    probBySPrime=probBySPrime+self.transitionFunction(s, a, sPrime)*b[s]
                probByO=probByO+probBySPrime*self.observationFunction(sPrime, a, o)
            prob=prob+probByO*(self.se(b,a,o)==bPrime)
        return prob
        
        
    def expect(self, xDistribution, function):
        expectation=sum([function(x)*px for x, px in xDistribution.items()])
        return expectation
        
    def se(self, b,a,o):
        bPrimeUnormalized={sPrime: self.observationFunction(sPrime, a, o)*self.expect(b, lambda s: self.transitionFunction(s, a, sPrime)) for sPrime in b}
        alpha=sum(bPrimeUnormalized.values())
        if alpha==0:
            return np.NaN
        bPrime={sPrime: bSPrimeUnormalized/alpha for sPrime, bSPrimeUnormalized in bPrimeUnormalized.items()}
        return bPrime   





class SampleBeliefReward(object):
    
    def __init__(self, transitionFunction, rewardFunction):
        self.transitionFunction=transitionFunction
        self.rewardFunction=rewardFunction
        
    def __call__(self, b, a):
        s=np.random.choice(list(b.keys()), p=list(b.values()))
        sPrimeProb={sPrime: self.transitionFunction(s, a, sPrime) for sPrime in b.keys()}
        sPrime=np.random.choice(list(sPrimeProb.keys()), p=list(sPrimeProb.values()))
        r=self.rewardFunction(s, a, sPrime)
        return r


class BeliefReward(object):
    
    def __init__(self, transitionFunction, rewardFunction, observationFunction, observationSpace):
        self.transitionFunction=transitionFunction
        self.rewardFunction=rewardFunction
        self.observationFunction=observationFunction
        self.observationSpace=observationSpace

    def __call__(self, b, a, bPrime):
        reward=0
        stateSpace=list(b.keys())
        for o in self.observationSpace:
            rewardByO=0
            for s in stateSpace:
                numerator=0
                denominator=0
                for sPrime in stateSpace:
                    numerator=numerator+self.transitionFunction(s, a, sPrime)*self.observationFunction(sPrime, a, o)*self.rewardFunction(s, a, sPrime)
                    denominator=denominator+self.transitionFunction(s, a, sPrime)*self.observationFunction(sPrime, a, o)
                if denominator!=0:
                    rewardByO=rewardByO+numerator/denominator*b[s]
                else:
                    rewardByO=rewardByO
            reward=reward+rewardByO*(self.se(b,a,o)==bPrime)
        return reward
    
    def expect(self, xDistribution, function):
        expectation=sum([function(x)*px for x, px in xDistribution.items()])
        return expectation
    
    def se(self, b,a,o):
        bPrimeUnormalized={sPrime: self.observationFunction(sPrime, a, o)*self.expect(b, lambda s: self.transitionFunction(s, a, sPrime)) for sPrime in b}
        alpha=sum(bPrimeUnormalized.values())
        if alpha==0:
            return np.NaN
        bPrime={sPrime: bSPrimeUnormalized/alpha for sPrime, bSPrimeUnormalized in bPrimeUnormalized.items()}
        return bPrime



class TigerTransition():
    def __init__(self):
        self.transitionMatrix = {
            ('listen', 'tiger-left', 'tiger-left'): 1.0,
            ('listen', 'tiger-left', 'tiger-right'): 0.0,
            ('listen', 'tiger-right', 'tiger-left'): 0.0,
            ('listen', 'tiger-right', 'tiger-right'): 1.0,

            ('open-left', 'tiger-left', 'tiger-left'): 0.5,
            ('open-left', 'tiger-left', 'tiger-right'): 0.5,
            ('open-left', 'tiger-right', 'tiger-left'): 0.5,
            ('open-left', 'tiger-right', 'tiger-right'): 0.5,

            ('open-right', 'tiger-left', 'tiger-left'): 0.5,
            ('open-right', 'tiger-left', 'tiger-right'): 0.5,
            ('open-right', 'tiger-right', 'tiger-left'): 0.5,
            ('open-right', 'tiger-right', 'tiger-right'): 0.5
        }

    def __call__(self, state, action, nextState):
        nextStateProb = self.transitionMatrix.get((action, state, nextState), 0.0)
        return nextStateProb


class TigerReward():
    def __init__(self, rewardParam):
        self.rewardMatrix = {
            ('listen', 'tiger-left'): rewardParam['listen_cost'],
            ('listen', 'tiger-right'): rewardParam['listen_cost'],

            ('open-left', 'tiger-left'): rewardParam['open_incorrect_cost'],
            ('open-left', 'tiger-right'): rewardParam['open_correct_reward'],

            ('open-right', 'tiger-left'): rewardParam['open_correct_reward'],
            ('open-right', 'tiger-right'): rewardParam['open_incorrect_cost']
        }

    def __call__(self, state, action, sPrime):
        rewardFixed = self.rewardMatrix.get((action, state), 0.0)
        return rewardFixed


class TigerObservation():
    def __init__(self, observationParam):
        self.observationMatrix = {
            ('listen', 'tiger-left', 'tiger-left'): observationParam['obs_correct_prob'],
            ('listen', 'tiger-left', 'tiger-right'): observationParam['obs_incorrect_prob'],
            ('listen', 'tiger-right', 'tiger-left'): observationParam['obs_incorrect_prob'],
            ('listen', 'tiger-right', 'tiger-right'): observationParam['obs_correct_prob'],

            ('open-left', 'tiger-left', 'Nothing'): 1,
            ('open-left', 'tiger-right', 'Nothing'): 1,
            ('open-right', 'tiger-left', 'Nothing'): 1,
            ('open-right', 'tiger-right', 'Nothing'): 1,
        }

    def __call__(self, state, action, observation):
        observationProb = self.observationMatrix.get((action, state, observation), 0.0)
        return observationProb

class SampleNextNode(object):
    def __init__(self, sampleNextBelief):
        self.sampleNextBelief=sampleNextBelief
        
    def __call__(self, node, a):
        b=node['b']
        depth=node['depth']
        bPrime=self.sampleNextBelief(b, a)
        newDepth=depth+1
        newNode={'b':bPrime, 'depth':newDepth}
        return newNode

class NodeReward(object):
    def __init__(self, beliefReward):
        self.beliefReward=beliefReward
        
    def __call__(self, node, a, nextNode):
        b=node['b']
        bPrime=nextNode['b']
        reward=self.beliefReward(b, a, bPrime)
        return reward

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


def main():
    
    rewardParam={'listen_cost':-1, 'open_incorrect_cost':-100, 'open_correct_reward':10}
    rewardFunction=TigerReward(rewardParam)
    
    observationParam={'obs_correct_prob':0.85, 'obs_incorrect_prob':0.15}
    observationFunction=TigerObservation(observationParam)
    
    transitionFunction=TigerTransition()
    
    observationSpace=['tiger-left', 'tiger-right', 'Nothing']
    
    sampleNextBelief=SampleNextBelief(transitionFunction, observationFunction, observationSpace)
    
    sampleBeliefReward=SampleBeliefReward(transitionFunction, rewardFunction)
    
    beliefTransition=BeliefTransition(transitionFunction, observationFunction, observationSpace)
    
    beliefReward=BeliefReward(transitionFunction, rewardFunction, observationFunction, observationSpace)
    
    b={'tiger-left':0.85, 'tiger-right':0.15}
    bPrime={'tiger-left':0.5, 'tiger-right':0.5}
    a='open-left'
    
    bPrime=sampleNextBelief(b, a)
    print(bPrime)
    
    node1={'b':{'tiger-left':0.85, 'tiger-right':0.15}, 'depth':10}
    sampleNextNode=SampleNextNode(sampleNextBelief)
    nodeReward=NodeReward(beliefReward)
    node2=sampleNextNode(node1, a)
    print(node2)
    print(nodeReward(node1, a, node2))

    
    print(beliefTransition(b, a, bPrime))
    print(beliefReward(b, a, bPrime))
    


if __name__=='__main__': 
    main()


