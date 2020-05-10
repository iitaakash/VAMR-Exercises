import numpy as np 
import glob
import cv2
import time


def ReadK(path = "data/K.txt"):
    K = np.eye(4)
    K[0:3,0:3] = np.loadtxt(path)
    return K

def ReadImage(path = "data/images_undistorted/img_0001.jpg"):
    img = cv2.imread(path)
    if len(img.shape) != 3:
        print("hello")
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    return img

def Read3dPts(path = "data/p_W_corners.txt"):
    data = np.loadtxt(path, delimiter = ",") * 0.01
    data_hom  =  np.hstack((data, np.ones((data.shape[0],1)))) 
    return data_hom

def Read2dPx(path = "data/detected_corners.txt"):
    data = np.loadtxt(path)
    n_pts = data.shape[0]
    px = data.reshape((n_pts,-1,2))
    # (210, 12, 2)
    ones = np.ones((px.shape[0], px.shape[1], 1))
    px_hom = np.concatenate((px,ones), axis = 2)
    # (210, 12, 3)
    return px_hom


# only works for this dataset
def ReadImages(path = "data/images_undistorted/", suf = ".jpg"):
    print("Loading image data")
    start = time.time()
    image_count = len(glob.glob(path + "*" + suf))
    images = []
    for i in range(image_count):
        image_path = path + "img_" + str(i+1).zfill(4) + suf
        img = ReadImage(image_path)
        images.append(img)
    images = np.array(images)
    end = time.time() - start 
    print("Loading image data took : {:0.2f} ".format(end))
    return images

