import cv2
import numpy as np 

def GetImage(path, scale):
    image = cv2.imread(path,0)
    X,Y = image.shape[1]*scale, image.shape[0]*scale
    result = cv2.resize(image,(int(X),int(Y)))#.astype(np.float)
    return result