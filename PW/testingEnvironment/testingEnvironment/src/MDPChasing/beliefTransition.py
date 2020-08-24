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




def main():
    
    rewardParam={'listen_cost':-1, 'open_incorrect_cost':-100, 'open_correct_reward':10}
    rewardFunction=TigerReward(rewardParam)
    
    observationParam={'obs_correct_prob':1, 'obs_incorrect_prob':0}
    observationFunction=TigerObservation(observationParam)
    
    transitionFunction=TigerTransition()
    
    observationSpace=['tiger-left', 'tiger-right', 'Nothing']
    
    sampleNextBelief=SampleNextBelief(transitionFunction, observationFunction, observationSpace)
    
    sampleBeliefReward=SampleBeliefReward(transitionFunction, rewardFunction)
    
    b={'tiger-left':0, 'tiger-right':1}
    a='open-left'
    
    bPrime=sampleNextBelief(b, a)
    print(bPrime)
    
    r=sampleBeliefReward(b, a)
    print(r)
    
if __name__ == '__main__':
    main()
   
    
    
    










