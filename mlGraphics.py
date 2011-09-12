"""
Some useful graphics functions
"""

import util
import binary

from numpy import *
from pylab import *

def plotLinearClassifier(h, X, Y):
    """
    Draw the current decision boundary, margin and data
    """

    if type(h.weights) == ndarray:
        nx,mx,ny,my = axis()

        # find the point y such that 
        #   nx*w[0] + y*w[1] == 0
        #   y = -(nx*w[0]) / w[1]
        nx_y = -(nx * h.weights[0]) / h.weights[1]
        mx_y = -(mx * h.weights[0]) / h.weights[1]

        # find the point x such that
        #   x*w[0] + ny*w[1] + b == 0
        #   x = -(ny*w[1] + b) / w[0]
        ny_x = -(ny * h.weights[1]) / h.weights[0]
        my_x = -(my * h.weights[1]) / h.weights[0]

        plot(X[Y>=0.5,0], X[Y>=0.5,1], 'b+',
             X[Y< 0.5,0], X[Y< 0.5,1], 'ro',
             [0.,h.weights[0]], [0.,h.weights[1]], 'g-',
             [nx,mx], [nx_y, mx_y], 'k-')
        legend(('positive', 'negative', 'weights', 'hyperplane'))
        nx,mx,ny,my = axis()

def runOnlineClassifier(h, X, Y):
    N,D = X.shape
    order = range(N)
    util.permute(order)
    plot(X[Y< 0.5,0], X[Y< 0.5,1], 'b+',
         X[Y>=0.5,0], X[Y>=0.5,1], 'ro')
    noStop = False
    for n in order:
        print (Y[n], X[n,:])
        h.nextExample(X[n,:], Y[n])
        hold(True)
        plot([X[n,0]], [X[n,1]], 'ys')
        hold(False)
        if not noStop:
            v = raw_input()
            if v == "q":
                noStop = True
        plotLinearClassifier(h, X, Y)

