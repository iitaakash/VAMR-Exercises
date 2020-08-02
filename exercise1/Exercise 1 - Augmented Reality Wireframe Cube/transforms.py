import numpy as np 
import math


def Pose2Tranf(pose):
    T = np.eye(4)
    T[0:3,3] = pose[3:6]

    w = pose[0:3]

    theta = np.linalg.norm(w)
    k = w
    if theta != 0:
        k = w / theta
    kx = np.array([[0, -k[2], k[1]], \
                   [k[2], 0, -k[0]], \
                   [-k[1], k[0], 0]])
    T[0:3,0:3] = np.eye(3) + np.sin(theta)*kx + (1.0 - np.cos(theta))*np.dot(kx,kx)
    return T

def World2Cam(T,points):
    return np.dot(T,points)

def Cam2Im(K,points):
    pix = np.dot(K,points)
    image_pts = pix[0:2,:] / pix[2,:]
    return image_pts

def Cam2ImDist(K,D,points):
    in_px = points[0:2,:] / points[2,:]
    r2 = in_px[0,:]*in_px[0,:] + in_px[1,:]*in_px[1,:]
    out_px = (1.0 + D[0]*r2 + D[1]*r2*r2 + D[2]*r2*r2*r2) * in_px
    hom_pts = np.vstack((out_px, np.ones((1,out_px.shape[1]))))
    out = np.dot(K[0:3,0:3],hom_pts)
    image_pts = out[0:2,:] / out[2,:]
    return image_pts

def Im2Pix(height,width,points):
    # frame_condition = ((points[0,:] < 0.0) +  (points[1,:] < 0.0) + (points[0,:] > width) + (points[1,:] > height)) > 0
    frame_condition = False
    points[:,frame_condition] = -1.0
    points = np.rint(points).astype(int)
    return np.transpose(points)

# points are 4xN numpy array
def World2Pix(K,T, height,width,points, D = np.array([0,0,0,0,0])):
    cam_points = World2Cam(T,points)
    img_pts = np.array([])
    if (D == np.array([0,0,0,0,0])).all():
        img_pts = Cam2Im(K,cam_points)
    else:
        img_pts = Cam2ImDist(K,D,cam_points)
    img_px = Im2Pix(height, width, img_pts)
    return img_px


def round2int(num):
    return int(round(num))