import numpy as np 
import glob
import cv2


def ReadK(path = "data/K.txt"):
    K = np.eye(4)
    K[0:3,0:3] = np.loadtxt(path)
    return K

def ReadD(path = "data/D.txt"):
    D = np.zeros((5,))
    data = np.loadtxt(path)
    if data.size != 5:
        for i,d in enumerate(data):
            D[i] = d
    return D


def ReadPoses(path = "data/poses.txt"):
    return np.loadtxt(path)


def ReadImage(path = "data/images_undistorted/img_0001.jpg"):
    img = cv2.imread(path)
    if len(img.shape) != 3:
        print("hello")
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    return img


# only works for this dataset
def ReadImages(path = "data/images/", suf = ".jpg"):
    image_count = len(glob.glob(path + "*" + suf))
    images = []
    for i in range(image_count):
        image_path = path + "img_" + str(i+1).zfill(4) + suf
        img = ReadImage(image_path)
        images.append(img)
    images = np.array(images)
    return images

