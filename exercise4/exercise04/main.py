from data import *
# from Sift import *
import matplotlib.pyplot as plt
import cv2



num_scales = 3 # Scales per octave.
num_octaves = 5 # Number of octaves.
sigma = 1.6
contrast_threshold = 0.04
image_file_1 = 'images/img_1.jpg'
image_file_2 = 'images/img_2.jpg'
rescale_factor = 0.2; # Rescaling of the original image for speed.

images = [GetImage(image_file_1, rescale_factor), GetImage(image_file_2, rescale_factor)]

def IsCenterMax(data):
    if data.shape != (3,3):
        return False
    middle = data[1,1]
    data[1,1] = 0
    if np.all(middle > np.max(data)):
        return True
    else:
        return False



for i, img in enumerate(images):
    # Write code to compute:
    # 1)    image pyramid. Number of images in the pyarmid equals
    #       'num_octaves'.
    # 2)    blurred images for each octave. Each octave contains
    #       'num_scales + 3' blurred images.
    # 3)    'num_scales + 2' difference of Gaussians for each octave.
    print(img.shape)
    img_key = img.copy()

    sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers = num_octaves)
    kp = sift.detect(img_key,None)

    img_key = cv2.drawKeypoints(img_key, kp, img_key)
   
    # keypoints_2 = Sift(img_key, num_scales, num_octaves, sigma, contrast_threshold)

    # img_key = img_key.astype(np.uint8)

    # for kp in keypoints:
    #     img_key = cv2.circle(img_key, (kp[1], kp[0]), radius= 3, color=(0, 0, 255), thickness=-1)


    cv2.imshow("image", img_key)
    cv2.waitKey(0)


    # 4)    Compute the keypoints with non-maximum suppression and
    #       discard candidates with the contrast threshold.
    # 5)    Given the blurred images and keypoints, compute the
    #       descriptors. Discard keypoints/descriptors that are too close
    #       to the boundary of the image. Hence, you will most likely
    #       lose some keypoints that you have computed earlier.

# Finally, match the descriptors using the function 'matchFeatures' and
# visualize the matches with the function 'showMatchedFeatures'.
# If you want, you can also implement the matching procedure yourself using
# 'knnsearch'.