from numpy import *
from pylab import *
import util, datasets, runClassifier, binary
import dt
import dumbClassifiers

X = datasets.TennisData.X
Y = datasets.TennisData.Y
data = datasets.TennisData

# ----- for dt -----
# maxD = 1;
# reload(dt)
# h = dt.DT({'maxDepth': maxD})
# h.train(X,Y)

# ----- for KNN.py -----
import knn
# eps ball
eps = 0.5
runClassifier.trainTestSet(knn.KNN({'isKNN':False, 'eps':eps}), datasets.TennisData)

runClassifier.trainTestSet(knn.KNN({'isKNN':False, 'eps':eps}), datasets.CFTookAI)


K = 1
runClassifier.trainTestSet(knn.KNN({'isKNN':True, 'K':K}), datasets.TennisData)

runClassifier.trainTestSet(knn.KNN({'isKNN':True, 'K':K}), datasets.CFTookAI)

# curves
curveKNN = runClassifier.hyperparamCurveSet(knn.KNN({'isKNN':True, 'K':1}), 'K', range(0,10), datasets.CFTookAI)

runClassifier.plotCurve('KNN on AI: K=[0:10]', curveKNN)

curveEps = runClassifier.hyperparamCurveSet(knn.KNN({'isKNN':False, 'eps':0.5}), 'eps',arange(1,10,0.5), datasets.CFTookAI)
runClassifier.plotCurve('epsilon on AI: eps=[1:0.5:10]', curveEps)

learningCurve = runClassifier.learningCurveSet(knn.KNN({'isKNN':True, 'K':5}), datasets.CFTookAI)
runClassifier.plotCurve('KNN on AI: K=5', learningCurve)
