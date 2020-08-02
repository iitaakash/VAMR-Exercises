from data import *
from pnp import *
import cv2
from plot3d import *
import time


# read data
K = ReadK()
images = ReadImages()
points3d = Read3dPts()
# 12,4
points2d = Read2dPx()
# N,12,3

# process data
R = []
t = []

poses = []
for i,im in enumerate(images):
    img = im.copy()

    start = time.time()
    M = GetCameraPosePNP(points3d, points2d[i] , K)
    R.append(M[:,0:3])
    t.append(M[:,3])
    pts = ReprojectPoints(points3d,M,K)
    end = time.time() - start
    # print("it took : {:0.6f}".format(end))

    for pt in pts:
        img = cv2.circle(img, (int(pt[0]), int(pt[1])), radius=3, color=(0, 0, 255), thickness=-1)

    cv2.imshow("image", img)
    k = cv2.waitKey(30)
    if k == 32:
        break


# R = np.array(R).reshape((-1,3,3))
# t = np.array(t).reshape((-1,3))
# plotTrajectory3D(t, R, points3d)