import numpy as np 
import cv2
from PoseEstimation import *
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

img = cv2.imread("data/0001.jpg")
img_2 = cv2.imread("data/0002.jpg")

K = np.array([1379.74, 0, 760.35, 0, 1382.08, 503.41, 0, 0, 1 ]).reshape((3,3))

## Load outlier-free point correspondences

p1 = np.loadtxt("data/matches0001.txt")
p2 = np.loadtxt("data/matches0002.txt")
# 84 points 

ones = np.ones((1, p1.shape[1]))
p1 = np.vstack((p1,ones))
p2 = np.vstack((p2,ones))

## Estimate the essential matrix E using the 8-point algorithm

E = EstimateEssentialMatrix(p1, p2, K, K)

## Extract the relative camera positions (R,T) from the essential matrix

# Obtain extrinsic parameters (R,t) from E
(Rots,u3) = DecomposeEssentialMatrix(E)

# Disambiguate among the four possible configurations
(R_C2_W,T_C2_W) = DisambiguateRelativePose(Rots,u3,p1,p2,K,K)

# # Triangulate a point cloud using the final transformation (R,T)
M1 = np.dot(K, np.eye(3,4))
M2 = np.dot(K, np.concatenate((R_C2_W , T_C2_W.reshape((3,1))), axis = 1))
P = LinearTriangulation(p1,p2,M1,M2)


fig = pyplot.figure()
ax = Axes3D(fig)

ax.scatter(P[0,:], P[1,:], P[2,:])
ax.scatter(0, 0, 0, "ro")
ax.scatter(T_C2_W[0], T_C2_W[1], T_C2_W[2], "ro")
pyplot.show()