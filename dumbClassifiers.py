"""
In dumbClassifiers.py, we implement the world's simplest classifiers:
  1) Always predict +1
  2) Always predict the most frequent label from the training data
  3) Just use the sign of the first feature to decide on label
"""

from binary import *
from numpy  import *

import util

class AlwaysPredictOne(BinaryClassifier):
    """
    This defines the classifier that always predicts +1.
    """

    def __init__(self, opts):
        """
        do nothing
        """

    def online(self):
        return False

    def __repr__(self):
        return "AlwaysPredictOne"

    def predict(self, X):
        return 1       # return our constant prediction

    def train(self, X, Y):
        """
        do nothing
        """


class AlwaysPredictMostFrequent(BinaryClassifier):
    """
    This defines the classifier that always predicts the
    most frequent label from the training data.
    """

    def __init__(self, opts):
        """
        if we haven't been trained, assume most frequent class is +1
        """
        self.mostFrequentClass = 1

    def online(self):
        return False

    def __repr__(self):
        return "AlwaysPredictMostFrequent(%d)" % self.mostFrequentClass

    def predict(self, X):
		return self.mostFrequentClass

    def train(self, X, Y):
		dataSum = Y.sum()
		if dataSum >= 0:
			self.mostFrequentClass = 1
		else:
			self.mostFrequentClass = -1

class FirstFeatureClassifier(BinaryClassifier):
    """
    This defines the classifier that always predicts on the basis of
    the first feature only.  In particular, we maintain two
    predictors: one for when the first feature is >0, one for when the
    first feature is <= 0.
    """

    def __init__(self, opts):
        """
        if we haven't been trained, always return 1
        """
        self.classForPos = 1    # what class should we return if X[0] >  0
        self.classForNeg = 1    # what class should we return if X[0] <= 0

    def online(self):
        return False

    def __repr__(self):
        return "FirstFeatureClassifier(%d,%d)" % (self.classForPos, self.classForNeg)

    def predict(self, X):
		if X[0] > 0:
			return self.classForPos
		else:
			return self.classForNeg

    def train(self, X, Y):
		index = 0
		posSum = 0
		negSum = 0
		for featureV in X:
			if featureV[0] > 0:
				posSum += Y[index]
			else:
				negSum += Y[index]
			index += 1
		if posSum >= 0:
			self.classForPos = 1
		else:
			self.classForPos = -1
		if negSum >= 0:
			self.classForNeg = 1
		else:
			self.classForNeg = -1
