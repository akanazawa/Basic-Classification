from numpy import *
from pylab import *
import util, datasets, runClassifier, binary
import dt
import dumbClassifiers

X = datasets.TennisData.X
Y = datasets.TennisData.Y
data = datasets.TennisData

# for dt
# maxD = 1;
# reload(dt)
# h = dt.DT({'maxDepth': maxD})
# h.train(X,Y)
