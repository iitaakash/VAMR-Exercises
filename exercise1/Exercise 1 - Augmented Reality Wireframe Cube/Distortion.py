import math
import numpy as np
import time
from transforms import *


# Class for precomputing distortions
class Distortion:

    def __init__(self, height, width, K = np.eye(3), D = np.zeros((5,))):
        self.height = int(height)
        self.width = int(width)
        self.x_pts = None
        self.y_pts = None
        self.mul_vals = None
        self.mamb = None
        self.mab = None
        self.amb = None
        self.ab = None
        self.x = None
        self.y = None
        self.K = K
        self.D = D
        self.Kinv = None
        self.undist = np.zeros((self.height, self.width, 3))
        self.undist = self.undist.astype(np.uint8)
        self.tot_size =  self.height * self.width 
    

    # shaving off 70 ms with this precomputation of params.
    def InitDistortionMap(self):
        print("Building distortion map")
        start = time.time()
        self.undist = np.zeros((self.height, self.width, 3))
        self.undist = self.undist.astype(np.uint8)
        self.tot_size =  self.height * self.width 
        self.x_pts,self.y_pts = np.meshgrid(np.arange(self.height), np.arange(self.width))
        self.x_pts = self.x_pts.flatten()
        self.y_pts = self.y_pts.flatten()
        px = np.vstack((self.y_pts, self.x_pts, np.ones(self.x_pts.shape) ,np.ones(self.x_pts.shape)))
        self.Kinv = np.linalg.inv(self.K)
        points = np.dot(self.Kinv , px)
        pix = Cam2ImDist(self.K,self.D,points)
        X = pix
        Xf = np.floor(X).astype(int)
        A = X - Xf
        self.x = Xf[1,:]
        self.y = Xf[0,:]
        a = A[1,:,np.newaxis]
        b = A[0,:,np.newaxis]
        ma = (1.0 - a)
        mb = (1.0 - b)
        self.mamb = ma * mb
        self.mab = ma * b
        self.amb = a * mb
        self.ab = a * b
        self.mul_vals = np.vstack((self.mamb, self.amb, self.ab, self.mab))
        end = time.time() - start
        print("Building distortion map took : {:0.3f}s ".format(end))

    

    def Undistort(self, image, bilenear = True):
        if bilenear:
            img_data  = np.vstack((image[self.x, self.y,:] , image[self.x + 1, self.y,:] , image[self.x + 1, self.y + 1,:] , image[self.x, self.y + 1,:]))
            self.undist[self.x_pts,self.y_pts,:] = np.sum((img_data * self.mul_vals).reshape((4,self.tot_size,-1)),axis=0)
            # self.undist[self.x_pts,self.y_pts,:] = image[self.x, self.y,:] * (self.mamb) + \
            #                                        image[self.x + 1, self.y,:] * (self.amb) + \
            #                                        image[self.x + 1, self.y + 1,:] * (self.ab) + \
            #                                        image[self.x, self.y + 1,:] * (self.mab)
        else:
            # todo: implement round instead of floor
            self.undist[self.x_pts,self.y_pts,:] = image[self.x, self.y,:]
        return self.undist

def UndistortImage(image, K, Kinv, D = np.array([0,0,0,0,0]), bilenear = True):
    undist = np.zeros(image.shape)
    for i in range(undist.shape[0]):
        for j in range(undist.shape[1]):
            point = np.dot(Kinv , np.array([j,i,1,1]).reshape(4,1))
            pix = Cam2ImDist(K,D,point)
            if bilenear:
                pix = np.transpose(pix)
                x = pix[0][1]
                y = pix[0][0]
                xf = int(math.floor(x))
                yf = int(math.floor(y))
                a = x - xf
                b = y - yf
                if xf >= 0 and yf >=0 and xf+1 < image.shape[0] and yf+1 < image.shape[1]:
                    undist[i,j,:] = (1-b) * (image[xf,yf]*(1-a) + \
                                    image[xf+1,yf]*(a)) + (b) * (image[xf+1,yf+1]*(a) + \
                                    image[xf,yf+1]*(1-a))
            else:
                # simple rounding off
                pix = Im2Pix(0,0,pix)
                undist[i,j,:] = image[pix[0][1], pix[0][0],:]
    undist = undist.astype(np.uint8)
    return undist


def UndistortImageVec(image, K, Kinv, D = np.array([0,0,0,0,0]) , bilenear = False):
    undist = np.zeros(image.shape)
    height = undist.shape[0]
    width  = undist.shape[1]
    x_pts,y_pts = np.meshgrid(np.arange(height), np.arange(width))
    x_pts = x_pts.flatten()
    y_pts = y_pts.flatten()
    px = np.vstack((y_pts, x_pts, np.ones(x_pts.shape) ,np.ones(x_pts.shape)))
    points = np.dot(Kinv , px)
    pix = Cam2ImDist(K,D,points)
    # we need a condition here
    if bilenear:
        X = pix
        Xf = np.floor(X).astype(int)
        A = X - Xf
        x = Xf[1,:]
        y = Xf[0,:]
        a = A[1,:,np.newaxis]
        b = A[0,:,np.newaxis]
        undist[x_pts,y_pts,:] = (1.0 - b) * (image[x, y,:] * (1.0 - a) + image[x + 1, y,:] * (a)) + \
                                (b) * (image[x + 1, y + 1,:] * (a) + image[x, y + 1,:] * (1.0 - a))
    else:
        pix = Im2Pix(0,0,pix)
        undist[x_pts,y_pts,:] = image[pix[:,1], pix[:,0],:]
    undist = undist.astype(np.uint8)
    return undist