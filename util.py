import sys
import inspect
import random
import pylab
from numpy import *

def raiseNotDefined():
  print "Method not implemented: %s" % inspect.stack()[1][3]    
  sys.exit(1)

def permute(a):
  """
  Randomly permute the elements in array a
  """
  for n in range(len(a)):
    m = int(pylab.rand() * (len(a) - n)) + n
    t = a[m]
    a[m] = a[n]
    a[n] = t
    
def splitTrainTest(X0, Y0, freqTest):
  """
  Split data in X0/Y0 into train/test data with freqTest
  frequency of test points
  """
  N,D = X0.shape
  isTest = zeros(N, dtype=bool)
  for n in range(0, N, freqTest):
    isTest[n] = True
  X   = X0[isTest==False, :]
  Y   = Y0[isTest==False]
  Xte = X0[isTest, :]
  Yte = Y0[isTest]

  return (X,Y,Xte,Yte)


def uniq(seq, idfun=None): 
  # order preserving
  if idfun is None:
    def idfun(x): return x
  seen = {}
  result = []
  for item in seq:
    marker = idfun(item)
    # in old Python versions:
    # if seen.has_key(marker)
    # but in new ones:
    if marker in seen: continue
    seen[marker] = 1
    result.append(item)
  return result

def mode(seq):
  if len(seq) == 0:
    return 1.
  else:
    cnt = {}
    for item in seq:
      if cnt.has_key(item):
        cnt[item] += 1
      else:
        cnt[item] = 1
    maxItem = seq[0]
    for item,c in cnt.iteritems():
      if c > cnt[maxItem]:
        maxItem = item
    return maxItem
