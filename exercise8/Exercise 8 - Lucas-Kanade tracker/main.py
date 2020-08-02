import numpy as np
import matplotlib.pyplot as plt
import cv2

from Tracker import *

# Part 1: Warping images
# I_R = cv2.imread('data/000000.png',0);

# plt.figure(1)

# plt.subplot(2,2,1)
# plt.imshow(I_R)
# plt.title('Reference image')

# plt.subplot(2,2,2)
# W = getSimWarp(50, -30, 0, 1)
# plt.imshow(warpImage(I_R, W))
# plt.title('Translation')

# plt.subplot(2,2,3)
# W = getSimWarp(0, 0, 10, 1)
# plt.imshow(warpImage(I_R, W))
# plt.title('Rotation around upper left corner')

# plt.subplot(2,2,4)
# W = getSimWarp(0, 0, 0, 0.5)
# plt.imshow(warpImage(I_R, W))
# plt.title('Zoom on upper left corner')

# plt.show()

# # ## Part 2: Warped patches and recovering a simple warp with brute force
# I_R = cv2.imread('data/000000.png',0)
# plt.figure(2)
# # Get and display template:
# plt.subplot(1, 2, 1)
# W0 = getSimWarp(0, 0, 0, 1)
# x_T = np.array([900, 291])
# r_T = 15
# template = getWarpedPatch(I_R, W0, x_T, r_T)
# plt.imshow(template, cmap='jet')
# plt.title('Template')

# print("hello")
# plt.subplot(1, 2, 2)
# W = getSimWarp(10, 6, 0, 1)
# I = warpImage(I_R, W)
# r_D = 20
# (dx, ssds) = trackBruteForce(I_R, I, x_T, r_T, r_D)
# plt.imshow(ssds, cmap='jet', interpolation='nearest')
# plt.title('SSDs')
# print("Displacement best explained by (dx, dy) = {}".format(dx))

# plt.show()

# ## Part 3: Recovering the warp with KLT
# I_R = cv2.imread('data/000000.png', 0)
# x_T = np.array([899,290])
# r_T = 15
# num_iters = 50
# W = getSimWarp(10, 6, 0, 1)
# I = warpImage(I_R, W)
# (W, p_hist) = trackKLT(I_R, I, x_T, r_T, num_iters)
# print("Point moved by {}".format(W[:,-1]))

# ## Part 4: Applying KLT to KITTI
# I_R = cv2.imread('data/000000.png',0)

# r_T = 15
# num_iters = 50

# width = int(I_R.shape[1]/4)
# height = int(I_R.shape[0]/4)

# I_R = cv2.resize(I_R,(width,height))

# img = cv2.cvtColor(I_R,cv2.COLOR_GRAY2RGB)

# keypoints_rc = np.loadtxt('data/keypoints.txt') / 4
# keypoints_rc = keypoints_rc.astype(np.int)

# keypoints = keypoints_rc[0:2, :]

# cv2.namedWindow("image", cv2.WINDOW_NORMAL)

# for kp in keypoints:
#     pt = (kp[1],kp[0])
#     img = cv2.circle(img, pt, radius= 1, color=(0, 0, 255), thickness=-1)

# cv2.imshow("image", img)
# cv2.waitKey(1)
# I_prev = I_R.copy()

# for i in range(1,20):
#     print(i)
#     I = cv2.imread('data/{}.png'.format(str(i).zfill(6)), 0)
#     I = cv2.resize(I,(width,height))
#     # cv2.imshow("image", I)
#     # cv2.waitKey(20)

#     dkp = np.zeros(keypoints.shape)
#     for j in range(keypoints.shape[0]):
#         pt = np.array([keypoints[j,1] , keypoints[j,0]])
#         (W, hist) = trackKLT(I_prev, I, pt, r_T, num_iters)
#         w = np.array([W[:, -1][1], W[:, -1][0]])
#         dkp[j, :] = w

#     kpold = keypoints
#     keypoints = (keypoints + dkp).astype(int)

#     img = cv2.cvtColor(I,cv2.COLOR_GRAY2RGB)

#     for kp in keypoints:
#         pt = (kp[1],kp[0])
#         img = cv2.circle(img, pt, radius= 1, color=(0, 0, 255), thickness=-1)

#     I_prev = I.copy()
#     cv2.imshow("image", img)
#     cv2.waitKey(1)

# %% Part 5: Outlier rejection with bidirectional error

I_R = cv2.imread('data/000000.png',0)

r_T = 15
num_iters = 50
llambda = 0.1

width = int(I_R.shape[1]/4)
height = int(I_R.shape[0]/4)

I_R = cv2.resize(I_R,(width,height))

img = cv2.cvtColor(I_R,cv2.COLOR_GRAY2RGB)

keypoints_rc = np.loadtxt('data/keypoints.txt') / 4
keypoints_rc = keypoints_rc.astype(np.int)

keypoints = keypoints_rc[300:305, :]

cv2.namedWindow("image", cv2.WINDOW_NORMAL)

for kp in keypoints:
    pt = (kp[1],kp[0])
    img = cv2.circle(img, pt, radius= 1, color=(0, 0, 255), thickness=-1)

cv2.imshow("image", img)
cv2.waitKey(1)
I_prev = I_R.copy()

for i in range(1,20):
    print(i)
    I = cv2.imread('data/{}.png'.format(str(i).zfill(6)), 0)
    I = cv2.resize(I,(width,height))
    # cv2.imshow("image", I)
    # cv2.waitKey(20)

    dkp = np.zeros(keypoints.shape)
    keep = np.ones(keypoints.shape[0]) > 0
    for j in range(keypoints.shape[0]):
        pt = np.array([keypoints[j,1] , keypoints[j,0]])
        (W, hist, keep[j]) = trackKLTRobustly(I_prev, I, pt, r_T, num_iters, llambda)
        w = np.array([W[:, -1][1], W[:, -1][0]])
        dkp[j, :] = w

    kpold = keypoints[keep]
    keypoints = (keypoints + dkp).astype(int)
    keypoints = keypoints[keep,:]

    img = cv2.cvtColor(I,cv2.COLOR_GRAY2RGB)

    for kp in keypoints:
        pt = (kp[1],kp[0])
        img = cv2.circle(img, pt, radius= 1, color=(0, 0, 255), thickness=-1)

    I_prev = I.copy()
    cv2.imshow("image", img)
    cv2.waitKey(1)