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
h.train(X,Y)
curve = runClassifier.learningCurveSet(dt.DT({'maxDepth': 5}), datasets.CFTookAI)
curveh = runClassifier.hyperparamCurveSet(dt.DT({'maxDepth': 5}), 'maxDepth', [1,2,3,4,5,6,7,8,9,10], datasets.CFTookAI)
# the maxDepth that does the best on test set is 5, we reach 1 on the training data with maxDepth=10
# with maxDepth=5, the features are:
with5 = dt.DT({'maxDepth':5})
with5.train(datasets.CFTookAI.X, datasets.CFTookAI.Y)
# CFTookAI.X is (400, 55), the 55 features.
# the top 5 features are (indices):
top5With5 = [1,44, 37, 54, 52];
#which are:
bestWith5_courses = datasets.CFTookAI.courseNames[top5With5]
#result: array(['introduction to information technology',
      #  'database management systems', 'complexity theory',
      #  'computational linguistics ii', 'advanced computer graphics'], 
      # dtype='|S62')
worst5With5 = [53,36,24,18,7]; #reading the first depth 4 feature breadth wise
worstWith5_coures = datasets.CFTookAI.courseNames[worst5With5]
#array(['computational geometry', 'analysis of algorithms',
      #  'computer graphics', 'computer and network security',
      #  'introduction to low-level programming concepts'], 
      # dtype='|S62')

#top 5 features with depth 10
with10 = dt.DT({'maxDepth':10})
with10.train(datasets.CFTookAI.X, datasets.CFTookAI.Y)
with10
top5with10 = [1,44,37, 54, 52]; # the same
worst5with10=[45,15,47,35,51]; # this is different
worstWith10_coures = datasets.CFTookAI.courseNames[worst5With10]
#result:array(['geographical information systems and spatial databases',
      #  'algorithms', 'neural modeling',
      #  'empirical research methods for computer science',
      #  'fundamentals of software testing'], 
      # dtype='|S62')
#just for log, the full tree with depth 10
# Branch 1 Depth:0
#   Branch 44 Depth:1
#     Branch 54 Depth:2
#       Branch 22 Depth:3
#         Branch 53 Depth:4
#           Branch 7 Depth:5
#             Branch 20 Depth:6
#               Leaf -1.0
#               Leaf 1.0
#             Branch 52 Depth:6
#               Leaf -1.0
#               Branch 51 Depth:7
#                 Leaf -1.0
#                 Branch 45 Depth:8
#                   Leaf 1.0
#                   Leaf -1.0
#           Branch 13 Depth:5
#             Branch 52 Depth:6
#               Branch 51 Depth:7
#                 Branch 15 Depth:8
#                   Leaf 1.0
#                   Leaf -1.0
#                 Leaf -1.0
#               Leaf -1.0
#             Branch 48 Depth:6
#               Leaf -1.0
#               Leaf 1.0
#         Branch 14 Depth:4
#           Branch 6 Depth:5
#             Branch 2 Depth:6
#               Leaf -1.0
#               Leaf 1.0
#             Branch 53 Depth:6
#               Leaf -1.0
#               Branch 37 Depth:7
#                 Leaf -1.0
#                 Leaf 1.0
#           Branch 53 Depth:5
#             Leaf 1.0
#             Branch 37 Depth:6
#               Leaf 1.0
#               Leaf -1.0
#       Branch 47 Depth:3
#         Branch 50 Depth:4
#           Branch 22 Depth:5
#             Branch 53 Depth:6
#               Leaf 1.0
#               Leaf -1.0
#             Branch 53 Depth:6
#               Branch 52 Depth:7
#                 Leaf 1.0
#                 Leaf -1.0
#               Leaf 1.0
#           Branch 43 Depth:5
#             Leaf -1.0
#             Branch 52 Depth:6
#               Leaf -1.0
#               Leaf 1.0
#         Branch 17 Depth:4
#           Branch 40 Depth:5
#             Branch 42 Depth:6
#               Leaf 1.0
#               Leaf -1.0
#             Leaf 1.0
#           Branch 53 Depth:5
#             Branch 52 Depth:6
#               Leaf -1.0
#               Branch 21 Depth:7
#                 Leaf 1.0
#                 Leaf -1.0
#             Leaf -1.0
#     Branch 52 Depth:2
#       Branch 7 Depth:3
#         Branch 36 Depth:4
#           Branch 8 Depth:5
#             Leaf -1.0
#             Leaf 1.0
#           Branch 54 Depth:5
#             Leaf 1.0
#             Branch 29 Depth:6
#               Leaf -1.0
#               Leaf 1.0
#         Branch 54 Depth:4
#           Branch 15 Depth:5
#             Branch 53 Depth:6
#               Branch 51 Depth:7
#                 Branch 47 Depth:8
#                   Leaf 1.0
#                   Leaf -1.0
#                 Leaf -1.0
#               Branch 41 Depth:7
#                 Leaf -1.0
#                 Leaf 1.0
#             Branch 50 Depth:6
#               Leaf -1.0
#               Leaf 1.0
#           Branch 43 Depth:5
#             Leaf 1.0
#             Branch 53 Depth:6
#               Branch 35 Depth:7
#                 Leaf -1.0
#                 Leaf 1.0
#               Leaf -1.0
#       Branch 54 Depth:3
#         Branch 5 Depth:4
#           Branch 53 Depth:5
#             Branch 45 Depth:6
#               Leaf 1.0
#               Branch 51 Depth:7
#                 Leaf -1.0
#                 Leaf 1.0
#             Leaf 1.0
#           Branch 48 Depth:5
#             Leaf -1.0
#             Branch 53 Depth:6
#               Branch 37 Depth:7
#                 Leaf -1.0
#                 Leaf 1.0
#               Leaf 1.0
#         Branch 53 Depth:4
#           Branch 51 Depth:5
#             Branch 23 Depth:6
#               Leaf 1.0
#               Leaf -1.0
#             Branch 50 Depth:6
#               Leaf 1.0
#               Branch 45 Depth:7
#                 Leaf -1.0
#                 Leaf 1.0
#           Branch 12 Depth:5
#             Leaf 1.0
#             Leaf -1.0
#   Branch 37 Depth:1
#     Branch 54 Depth:2
#       Branch 36 Depth:3
#         Branch 24 Depth:4
#           Branch 5 Depth:5
#             Branch 53 Depth:6
#               Leaf 1.0
#               Branch 30 Depth:7
#                 Leaf 1.0
#                 Leaf -1.0
#             Branch 51 Depth:6
#               Leaf -1.0
#               Leaf 1.0
#           Branch 45 Depth:5
#             Leaf -1.0
#             Branch 38 Depth:6
#               Leaf 1.0
#               Branch 34 Depth:7
#                 Leaf 1.0
#                 Leaf -1.0
#         Branch 53 Depth:4
#           Branch 52 Depth:5
#             Branch 51 Depth:6
#               Leaf 1.0
#               Branch 45 Depth:7
#                 Leaf 1.0
#                 Leaf -1.0
#             Branch 51 Depth:6
#               Leaf 1.0
#               Branch 18 Depth:7
#                 Leaf 1.0
#                 Leaf -1.0
#           Branch 27 Depth:5
#             Branch 52 Depth:6
#               Branch 40 Depth:7
#                 Leaf 1.0
#                 Leaf -1.0
#               Leaf 1.0
#             Branch 34 Depth:6
#               Leaf -1.0
#               Leaf 1.0
#       Branch 50 Depth:3
#         Branch 18 Depth:4
#           Branch 35 Depth:5
#             Leaf -1.0
#             Leaf 1.0
#           Branch 11 Depth:5
#             Branch 13 Depth:6
#               Leaf 1.0
#               Leaf -1.0
#             Leaf 1.0
#         Branch 53 Depth:4
#           Branch 44 Depth:5
#             Branch 40 Depth:6
#               Branch 52 Depth:7
#                 Leaf 1.0
#                 Branch 35 Depth:8
#                   Leaf 1.0
#                   Leaf -1.0
#               Leaf -1.0
#             Branch 47 Depth:6
#               Leaf 1.0
#               Leaf -1.0
#           Branch 52 Depth:5
#             Branch 30 Depth:6
#               Leaf 1.0
#               Leaf -1.0
#             Leaf 1.0
#     Branch 48 Depth:2
#       Branch 35 Depth:3
#         Branch 7 Depth:4
#           Branch 54 Depth:5
#             Leaf 1.0
#             Branch 53 Depth:6
#               Leaf -1.0
#               Branch 52 Depth:7
#                 Leaf 1.0
#                 Branch 51 Depth:8
#                   Branch 47 Depth:9
#                     Leaf -1.0
#                     Leaf 1.0
#                   Leaf -1.0
#           Branch 27 Depth:5
#             Leaf 1.0
#             Branch 23 Depth:6
#               Leaf -1.0
#               Leaf 1.0
#         Branch 54 Depth:4
#           Branch 53 Depth:5
#             Branch 52 Depth:6
#               Leaf -1.0
#               Branch 51 Depth:7
#                 Branch 38 Depth:8
#                   Leaf 1.0
#                   Leaf -1.0
#                 Leaf -1.0
#             Branch 50 Depth:6
#               Branch 36 Depth:7
#                 Leaf -1.0
#                 Leaf 1.0
#               Branch 33 Depth:7
#                 Leaf 1.0
#                 Leaf -1.0
#           Branch 27 Depth:5
#             Branch 43 Depth:6
#               Leaf 1.0
#               Leaf -1.0
#             Leaf -1.0
#       Branch 46 Depth:3
#         Branch 27 Depth:4
#           Leaf -1.0
#           Branch 35 Depth:5
#             Branch 28 Depth:6
#               Leaf 1.0
#               Leaf -1.0
#             Branch 54 Depth:6
#               Branch 30 Depth:7
#                 Leaf 1.0
#                 Leaf -1.0
#               Leaf 1.0
#         Branch 54 Depth:4
#           Branch 23 Depth:5
#             Leaf 1.0
#             Branch 44 Depth:6
#               Leaf 1.0
#               Branch 33 Depth:7
#                 Leaf -1.0
#                 Leaf 1.0
#           Branch 53 Depth:5
#             Leaf 1.0
#             Branch 51 Depth:6
#               Branch 22 Depth:7
#                 Leaf 1.0
#                 Leaf -1.0
#               Leaf 1.0

# for wu 4
CGDT = dt.DT({'maxDepth': 3})
CGDT.train(datasets.CFTookCG.X, datasets.CFTookCG.Y)



# ----- for KNN.py -----
# import knn
# # eps ball
# eps = 0.5
# runClassifier.trainTestSet(knn.KNN({'isKNN':False, 'eps':eps}), datasets.TennisData)

# runClassifier.trainTestSet(knn.KNN({'isKNN':False, 'eps':eps}), datasets.CFTookAI)


# K = 1
# runClassifier.trainTestSet(knn.KNN({'isKNN':True, 'K':K}), datasets.TennisData)

# runClassifier.trainTestSet(knn.KNN({'isKNN':True, 'K':K}), datasets.CFTookAI)

# # curves
# curveKNN = runClassifier.hyperparamCurveSet(knn.KNN({'isKNN':True, 'K':1}), 'K', range(0,10), datasets.CFTookAI)

# runClassifier.plotCurve('KNN on AI: K=[0:10]', curveKNN)

# curveEps = runClassifier.hyperparamCurveSet(knn.KNN({'isKNN':False, 'eps':0.5}), 'eps',arange(1,10,0.5), datasets.CFTookAI)
# runClassifier.plotCurve('epsilon on AI: eps=[1:0.5:10]', curveEps)

# learningCurve = runClassifier.learningCurveSet(knn.KNN({'isKNN':True, 'K':5}), datasets.CFTookAI)
# runClassifier.plotCurve('KNN on AI: K=5', learningCurve)

# # ----- Perceptron -----
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

# sorted_i = bestP.weights.argsort()
# bestW_inds = sorted_i[-5:]
# worstW_inds = sorted_i[0:5]
# bestWeights = bestP.weights[bestW_inds]
# worstWeights = bestP.weights[worstW_inds]

# #since each weight correspond to courseNames see what they correspond to:
# bestW_courses = datasets.CFTookAI.courseNames[bestW_inds]
# # response:
# # array(['database design', 'introduction to information technology',
# #        'computational methods', 'computer networks',
# #        'database management systems'], 
# #       dtype='|S62')

# worstW_courses = datasets.CFTookAI.courseNames[worstW_inds]
# response:
# array(['object-oriented programming i',
#        'introduction to human-computer interaction',
#        'program analysis and understanding', 'honors seminar',
#        'image processing'], 
