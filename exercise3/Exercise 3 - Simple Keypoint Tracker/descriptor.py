import numpy as np 
import cv2


def describeKeypoints(img, keypoints, r = 9):
    # Returns a (2r+1)^2xN matrix of image patch vectors based on image
    # img and a Nx2 matrix containing the keypoint coordinates.
    # r is the patch "radius".
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    patch_size = (2*r + 1)**2
    descriptors = np.zeros((keypoints.shape[0],patch_size))

    img_temp = np.pad(img, r ,'constant')

    for i,kp in enumerate(keypoints):
        x = kp + r
        descriptors[i,:] = img_temp[x[0]-r : x[0]+r+1, x[1]-r : x[1]+r+1].flatten()

    return descriptors