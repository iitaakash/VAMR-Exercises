import numpy as np 
import cv2
from corner import shi_tomasi,harris
from ptselect import selectKeypoints
from descriptor import describeKeypoints
import matplotlib.pyplot as plt
from matchdes import *



corner_patch_size = 9
harris_kappa = 0.08
num_keypoints = 200
nonmaximum_supression_radius = 8
descriptor_radius = 9
match_lambda = 4


img = cv2.imread('data/000000.png',1)

## Part 1 - Calculate Corner Response Functions

# Shi-Tomasi
shi_tomasi_scores = shi_tomasi(img, patch_size = corner_patch_size)

#Harris
harris_scores = harris(img, corner_patch_size, harris_kappa)


plt.figure()
plt.subplot(221)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)

plt.subplot(222)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)


plt.subplot(223)
plt.title('Shi-Tomasi Scores')
plt.imshow(shi_tomasi_scores, cmap='jet')

plt.subplot(224)
plt.title('Harris Scores')
plt.imshow(harris_scores, cmap='jet')

plt.show()


####################################################################################################################

# Part 2 - Select keypoints

keypoints = selectKeypoints(harris_scores, num_keypoints, nonmaximum_supression_radius)
img_key = img.copy()

for kp in keypoints:
    img_key = cv2.circle(img_key, (kp[1], kp[0]), radius= 3, color=(0, 0, 255), thickness=-1)


cv2.imshow("image", img_key)
cv2.waitKey(0)

####################################################################################################################

# Part 3 - Describe keypoints and show 16 strongest keypoint descriptors

descriptors = describeKeypoints(img, keypoints, descriptor_radius)

plt.figure(3)
for i in range(16):
    plt.subplot(4, 4, i+1)
    patch_size = 2 * descriptor_radius + 1
    cur_des = descriptors[i,:].reshape((patch_size,patch_size))
    plt.imshow(cur_des, cmap='jet')

plt.show()

####################################################################################################################


# Part 4 - Match descriptors between first two images
img_2 = cv2.imread('data/000001.png',1)
harris_scores_2 = harris(img_2, corner_patch_size, harris_kappa)
keypoints_2 = selectKeypoints(harris_scores_2, num_keypoints, nonmaximum_supression_radius)
descriptors_2 = describeKeypoints(img_2, keypoints_2, descriptor_radius)

matches = matchDescriptors(descriptors_2, descriptors, match_lambda)

img_match = img_2.copy()

for i, mch in enumerate(matches):
    if mch == -1:
        continue
    pt1 = (keypoints_2[i,1], keypoints_2[i,0])
    pt2 = (keypoints[mch,1], keypoints[mch,0])
    img_match = cv2.circle(img_match, pt1, radius= 3, color=(0, 0, 255), thickness=-1)
    img_match = cv2.circle(img_match, pt2, radius= 3, color=(255, 0, 0), thickness=-1)
    img_match = cv2.line(img_match, pt1, pt2, (0, 255, 0), thickness=2)



cv2.imshow("image", img_match)
cv2.waitKey(0)



####################################################################################################################
prev_kp = None
prev_desc = None
for i in range(200):
    img = cv2.imread('data/{}.png'.format(str(i).zfill(6)), 1)
    
    scores = harris(img, corner_patch_size, harris_kappa)
    kp = selectKeypoints(scores, num_keypoints, nonmaximum_supression_radius)
    desc = describeKeypoints(img, kp, descriptor_radius)
    if i > 0:
        matches = matchDescriptors(desc, prev_desc, match_lambda)

        for i, mch in enumerate(matches):
            if mch == -1:
                continue
            pt1 = (kp[i,1], kp[i,0])
            pt2 = (prev_kp[mch,1], prev_kp[mch,0])
            img = cv2.circle(img, pt1, radius= 6, color=(0, 0, 255), thickness=-1)
            img = cv2.circle(img, pt2, radius= 6, color=(0, 0, 255), thickness=-1)
            img = cv2.line(img, pt1, pt2, (0, 255, 0), thickness=2)

    cv2.imshow("image",img)
    k = cv2.waitKey(10)
    if k == 32:
        break
    prev_kp = kp
    prev_desc = desc