import numpy as np
from PoseEstimation import *

N = 1000
X = np.random.normal(size=(4,N))

X[2, :] = X[2, :] * 5 + 10
X[3, :] = 1

M1 = np.array([500, 0, 320, 0,0 ,500 ,240, 0, 0, 0, 1, 0]).reshape((3,4))

M2 = np.array([500, 0, 320, -100, 0, 500, 240, 0, 0, 0, 1, 0]).reshape((3,4))
				
p1 = np.dot(M1, X)
p2 = np.dot(M2, X)

X_est = LinearTriangulation(p1,p2,M1,M2)
error = np.mean(np.abs(X_est-X), axis = 1)
print('P_est-P=\n {}'.format(error))
