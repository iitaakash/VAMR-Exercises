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

def Cam2ImDist(K,D,points):
    in_px = points[0:2,:] / points[2,:]
    r2 = in_px[0,:]*in_px[0,:] + in_px[1,:]*in_px[1,:]
    out_px = (1.0 + D[0]*r2 + D[1]*r2*r2 + D[2]*r2*r2*r2) * in_px
    hom_pts = np.vstack((out_px, np.ones((1,out_px.shape[1]))))
    out = np.dot(K[0:3,0:3],hom_pts)
    return out

def Im2Pix(height,width,points):
    pixels = points[0:2,:] / points[2,:]
    # frame_condition = ((pixels[0,:] < 0.0) +  (pixels[1,:] < 0.0) + (pixels[0,:] > width) + (pixels[1,:] > height)) > 0
    frame_condition = False
    pixels[:,frame_condition] = -1.0
    pixels = np.rint(pixels).astype(int)
    return np.transpose(pixels)

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


def UndistortImage(image, K, Kinv, D = np.array([0,0,0,0,0]) ):
    undist = np.zeros(image.shape)
    for i in range(undist.shape[0]):
        for j in range(undist.shape[1]):
            point = np.dot(Kinv , np.array([j,i,1,1]).reshape(4,1))
            pix = Cam2ImDist(K,D,point)
            pix = Im2Pix(0,0,pix)
            undist[i,j,:] = image[pix[0][1], pix[0][0],:]
    undist = undist.astype(np.uint8)
    return undist


def UndistortImageVec(image, K, Kinv, D = np.array([0,0,0,0,0]) ):
    undist = np.zeros(image.shape)
    height = undist.shape[0]
    width  = undist.shape[1]
    x_pts,y_pts = np.meshgrid(np.arange(height), np.arange(width))
    x_pts = x_pts.flatten()
    y_pts = y_pts.flatten()
    px = np.vstack((y_pts, x_pts, np.ones(x_pts.shape) ,np.ones(x_pts.shape)))
    points = np.dot(Kinv , px)
    pix = Cam2ImDist(K,D,points)
    pix = Im2Pix(0,0,pix)
    undist[x_pts,y_pts,:] = image[pix[:,1], pix[:,0],:]
    undist = undist.astype(np.uint8)
    return undist