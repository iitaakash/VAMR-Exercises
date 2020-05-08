import numpy as np 


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
    return np.dot(K,points)

def Im2Pix(height,width,points):
    pixels = points[0:2,:] / points[2,:]
    # frame_condition = ((pixels[0,:] < 0.0) +  (pixels[1,:] < 0.0) + (pixels[0,:] > width) + (pixels[1,:] > height)) > 0
    frame_condition = False
    pixels[:,frame_condition] = -1.0
    pixels = np.rint(pixels).astype(int)
    return np.transpose(pixels)

# points are 4xN numpy array
def World2Pix(K,T,height,width,points):
    cam_points = World2Cam(T,points)
    img_pts = Cam2Im(K,cam_points)
    img_px = Im2Pix(height, width, img_pts)
    return img_px