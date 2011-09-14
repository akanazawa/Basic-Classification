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

# ----- Perceptron -----
import perceptron
epoch = 1;

runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch':epoch}), datasets.TennisData)

runClassifier.plotData(datasets.TwoDDiagonal.X, datasets.TwoDDiagonal.Y)
h = perceptron.Perceptron({'numEpoch':200})
h.train(datasets.TwoDDiagonal.X, datasets.TwoDDiagonal.Y)
runClassifier.plotClassifier(array([7.3, 18.9]), 0.0)

runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch':epoch}), datasets.CFTookAI)

epoches = arange(1,300,10)
curveP = runClassifier.hyperparamCurveSet(h, 'numEpoch', epoches, datasets.CFTookAI)
runClassifier.plotCurve('Perceptron on AI:#epoch=1:10:300', curveP);

bestSoFar_ind = curveP[1].argmax();
bestP = perceptron.Perceptron({'numEpoch':epoches[bestSoFar_ind]});
bestP.train(datasets.CFTookAI.X, datasets.CFTookAI.Y)

sorted_i = bestP.weights.argsort()
bestW_inds = sorted_i[-5:]
worstW_inds = sorted_i[0:5]
bestWeights = bestP.weights[bestW_inds]
worstWeights = bestP.weights[worstW_inds]

#since each weight correspond to courseNames see what they correspond to:
bestW_courses = datasets.CFTookAI.courseNames[bestW_inds]
# response:
# array(['database design', 'introduction to information technology',
#        'computational methods', 'computer networks',
#        'database management systems'], 
#       dtype='|S62')

worstW_courses = datasets.CFTookAI.courseNames[worstW_inds]
# response:
# array(['object-oriented programming i',
#        'introduction to human-computer interaction',
#        'program analysis and understanding', 'honors seminar',
#        'image processing'], 
