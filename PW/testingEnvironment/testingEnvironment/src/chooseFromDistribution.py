import numpy as np
import random


def maxFromDistribution(distribution):
    #print(distribution,type(distribution))
    hypotheses = list(distribution.keys())
    probs = list(distribution.values())
    maxIndices = np.argwhere(probs == np.max(probs)).flatten()
    selectedIndex = np.random.choice(maxIndices)
    selectedHypothesis = hypotheses[selectedIndex]
    return selectedHypothesis 


def sampleFromDistribution(distribution):
    print(distribution, type(distribution))
    hypotheses = list(distribution.keys())
    probs = list(distribution.values())
    normlizedProbs = [prob / sum(probs) for prob in probs]
    selectedIndex = list(np.random.multinomial(1, normlizedProbs)).index(1)
    selectedHypothesis = hypotheses[selectedIndex]
    return selectedHypothesis 

class SampleFromDistribution():
    def __init__(self, distribution):
        self.distribution = distribution

    def __call__(self, state):
        hypotheses = list(self.distribution.keys())
        probs = list(self.distribution.values())
        normlizedProbs = [prob / sum(probs) for prob in probs]
        selectedIndex = list(np.random.multinomial(1, normlizedProbs)).index(1)
        selectedHypothesis = hypotheses[selectedIndex]
        return selectedHypothesis 
