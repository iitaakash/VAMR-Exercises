import numpy as np
import cv2
from data import *
from transforms import *
from pointgen import *
import time



CHECK_SIZE = 0.04
WIDTH = 752.0
HEIGHT = 480.0

K = ReadK()
D = ReadD()
poses = ReadPoses()
imgs = ReadImages()


# ##################################################################################
# # read undistorted first image and overlay checkerboard
# path_undist = "data/images_undistorted/img_0001.jpg"
# img_undist = ReadImage(path_undist)

# img = img_undist.copy()

# T = Pose2Tranf(poses[0])

# pts = GetChessBoardPts(size = (6,9), res = CHECK_SIZE)
# pixels = World2Pix(K,T,HEIGHT,WIDTH,pts)
# for pt in pixels:
#     img = cv2.circle(img, tuple(pt), radius=3, color=(0, 0, 255), thickness=-1)

# cv2.imshow("perspective check", img)
# cv2.waitKey(0)

# ##################################################################################
# # read undistorted first image and overlay cube
# img = img_undist.copy()

# pts, edges = GetCubePoints(pos = np.array([0.02,0.06,0.0]), size = 0.02)
# pixels = World2Pix(K,T,HEIGHT,WIDTH,pts)

# for pt in edges:
#     pix1 = tuple(pixels[pt[0]])
#     pix2 = tuple(pixels[pt[1]])
#     if pix1[0] < 0 or pix1[1] < 0 or pix2[0] < 0 or pix2[1] < 0:
#         print("invalid pix")
#         continue
#     img = cv2.line(img, pix1, pix2, (0, 0, 255), thickness=2)

# cv2.imshow("cube draw", img)
# cv2.waitKey(0)

# ####################################################################################
# # read distorted images and overlay checkerboard

# for i, imag in enumerate(imgs):
#     img = imag.copy()
#     T = Pose2Tranf(poses[i])
#     pts = GetChessBoardPts(size = (6,9), res = CHECK_SIZE)
#     pixels = World2Pix(K,T,HEIGHT,WIDTH,pts,D)
#     for pt in pixels:
#         img = cv2.circle(img, tuple(pt), radius=3, color=(0, 0, 255), thickness=-1)

#     cv2.imshow("video distorted", img)
#     k = cv2.waitKey(15)
#     if k == 32 or k == 15:
#         break

####################################################################################
# read distorted images, undistort it and overlay cube

for i, imag in enumerate(imgs):
    img_dist = imag.copy()

    img = UndistortImageVec(img_dist,K,np.linalg.inv(K),D)
    T = Pose2Tranf(poses[i])
    
    pts, edges = GetCubePoints(pos = np.array([0.08,0.08,0.0]), size = 0.04)
    pixels = World2Pix(K,T,HEIGHT,WIDTH,pts)

    for pt in edges:
        pix1 = tuple(pixels[pt[0]])
        pix2 = tuple(pixels[pt[1]])
        if pix1[0] < 0 or pix1[1] < 0 or pix2[0] < 0 or pix2[1] < 0:
            print("invalid pix")
            continue
        img = cv2.line(img, pix1, pix2, (0, 0, 255), thickness=2)

    cv2.imshow("video undistorted", img)
    cv2.waitKey(1)

