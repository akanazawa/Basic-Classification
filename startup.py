from numpy import *
from pylab import *
import util, datasets, runClassifier, binary
import dt
import dumbClassifiers

X = datasets.TennisData.X
Y = datasets.TennisData.Y
data = datasets.TennisData

# ----- for dt -----
maxD = 1;
reload(dt)
h = dt.DT({'maxDepth': maxD})
curve = runClassifier.learningCurveSet(dt.DT({'maxDepth': 5}), datasets.CFTookAI)
curveh = runClassifier.hyperparamCurveSet(dt.DT({'maxDepth': 5}), 'maxDepth', [1,2,3,4,5,6,7,8,9,10], datasets.CFTookAI)
runClassifier.plotCurve('', curveh)

h = dt.DT({'maxDepth': 10})
h.train(datasets.CFTookAI.X, datasets.CFTookAI.Y);
# # the maxDepth that does the best on test set is 5, we reach 1 on the training data with maxDepth=10
# # with maxDepth=5, the features are:
 with5 = dt.DT({'maxDepth':5})
 with5.train(datasets.CFTookAI.X, datasets.CFTookAI.Y)
# # CFTookAI.X is (400, 55), the 55 features.
# # the top 5 features are (indices):
 top5With5 = [1,44, 37, 54, 52, 48];
# #which are:
 bestWith5_courses = datasets.CFTookAI.courseNames[top5With5]
# #result: array(['introduction to information technology',
#       #  'database management systems', 'complexity theory',
#       #  'computational linguistics ii', 'advanced computer graphics'], 
#       # dtype='|S62')
# worst5With5 = [53,36,24,18,7]; #reading the first depth 4 feature breadth wise
# worstWith5_coures = datasets.CFTookAI.courseNames[worst5With5]
# #array(['computational geometry', 'analysis of algorithms',
#       #  'computer graphics', 'computer and network security',
#       #  'introduction to low-level programming concepts'], 
#       # dtype='|S62')

# #top 5 features with depth 10
# with10 = dt.DT({'maxDepth':10})
# with10.train(datasets.CFTookAI.X, datasets.CFTookAI.Y)
# with10
# top5with10 = [1,44,37, 54, 52]; # the same
# worst5with10=[45,15,47,35,51]; # this is different
# worstWith10_coures = datasets.CFTookAI.courseNames[worst5With10]
#result:array(['geographical information systems and spatial databases',
      #  'algorithms', 'neural modeling',
      #  'empirical research methods for computer science',
      #  'fundamentals of software testing'], 
      # dtype='|S62')
# for wu 4
# CGDT = dt.DT({'maxDepth': 3})
# CGDT.train(datasets.CFTookCG.X, datasets.CFTookCG.Y)



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
curveKNN = runClassifier.hyperparamCurveSet(knn.KNN({'isKNN':True, 'K':1}), 'K', range(0,10), datasets.CFDataRatings)

runClassifier.plotCurve('KNN on AI: K=[0:10]', curveKNN)

curveEps = runClassifier.hyperparamCurveSet(knn.KNN({'isKNN':False, 'eps':0.5}), 'eps',arange(1,10,0.5), datasets.CFTookAI)
runClassifier.plotCurve('epsilon on AI: eps=[1:0.5:10]', curveEps)

learningCurve = runClassifier.learningCurveSet(knn.KNN({'isKNN':True, 'K':5}), datasets.CFTookAI)
runClassifier.plotCurve('KNN on AI: K=5', learningCurve)
learningCurveEps = runClassifier.learningCurveSet(knn.KNN({'isKNN':False, 'eps':5}), datasets.CFTookAI)
runClassifier.plotCurve('KNN on Eps: Eps=5', learningCurveEps)

# # ----- Perceptron -----
import perceptronAJ
epoch = 1;

runClassifier.trainTestSet(perceptronAJ.Perceptron({'numEpoch':epoch}), datasets.TennisData)

runClassifier.plotData(datasets.TwoDDiagonal.X, datasets.TwoDDiagonal.Y)
h = perceptronAJ.Perceptron({'numEpoch':200})
h.train(datasets.TwoDDiagonal.X, datasets.TwoDDiagonal.Y)
runClassifier.plotClassifier(array([7.3, 18.9]), 0.0)

runClassifier.trainTestSet(perceptronAJ.Perceptron({'numEpoch':epoch}), datasets.CFTookAI)

epoches = arange(1,100,1)
curveP = runClassifier.hyperparamCurveSet(h, 'numEpoch', epoches, datasets.CFTookAI)
runClassifier.plotCurve('Perceptron on AI:#epoch=1:1:300', curveP);

bestSoFar_ind = curveP[2].argmax();
bestP = perceptronAJ.Perceptron({'numEpoch':epoches[bestSoFar_ind]});
bestP.train(datasets.CFTookAI.X, datasets.CFTookAI.Y)

sorted_i = bestP.weights.argsort()
bestW_inds = sorted_i[-8:]
worstW_inds = sorted_i[0:8]
bestWeights = bestP.weights[bestW_inds]
worstWeights = bestP.weights[worstW_inds]

# #since each weight correspond to courseNames see what they correspond to:
bestW_courses = datasets.CFTookAI.courseNames[bestW_inds]
# # response:
# # array(['database design', 'introduction to information technology',
# #        'computational methods', 'computer networks',
# #        'database management systems'], 
# #       dtype='|S62')

worstW_courses = datasets.CFTookAI.courseNames[worstW_inds]
# response:
# array(['object-oriented programming i',
#        'introduction to human-computer interaction',
#        'program analysis and understanding', 'honors seminar',
#        'image processing'], 
