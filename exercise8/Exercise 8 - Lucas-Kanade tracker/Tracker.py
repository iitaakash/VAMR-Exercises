import numpy as np 
from scipy.signal import convolve2d
import cv2

def getSimWarp(dx, dy, alpha_deg, lam):
    alpha = np.radians(alpha_deg)
    c = np.cos(alpha)
    s = np.sin(alpha)
    W = lam * np.array([[c, -s, dx], [s, c, dy]])
    return W


def warpImage(img, W):
    out = np.zeros(img.shape)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            pt = np.dot(W , np.array([j,i,1]).reshape((3,1)))
            Xf = np.floor(pt).astype(int)
            A = pt - Xf
            x = Xf[1]
            y = Xf[0]
            a = A[1]
            b = A[0]
            if x + 1 >= out.shape[0] or y + 1 >= out.shape[1] or x <= -1 or y <= -1: 
                continue

            out[i,j] = (1.0 - b) * ((1.0 - a) * img[x,y] + (a) * img[x+1, y]) + \
                 (b) * (img[x + 1, y + 1] * (a) + img[x, y + 1] * (1.0 - a))
            
            
    return out.astype(np.uint8)


def getWarpedPatch(img, W, pt, r):
    out = np.zeros((2*r+1, 2*r+1))
    x_mid = r
    y_mid = r
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            point = np.array([[pt[0]],[pt[1]]]) + np.dot(W , np.array([j - x_mid ,i - y_mid,1]).reshape((3,1)))
            Xf = np.floor(point).astype(int)
            A = point - Xf
            x = Xf[1]
            y = Xf[0]
            a = A[1]
            b = A[0]
            if x + 1 >= img.shape[0] or y + 1 >= img.shape[1] or x <= -1 or y <= -1: 
                continue
            out[i,j] = (1.0 - b) * ((1.0 - a) * img[x,y] + (a) * img[x+1, y]) + \
                 (b) * (img[x + 1, y + 1] * (a) + img[x, y + 1] * (1.0 - a))
    return out.astype(np.uint8)


def trackBruteForce(I_R, I, x_T, r_T, r_D):
    # I_R: reference image, I: image to track point in, x_T: point to track,
    # expressed as [x y]=[col row], r_T: radius of patch to track, r_D: radius
    # of patch to search dx within; dx: translation that best explains where
    # x_T is in image I, ssds: SSDs for all values of dx within the patch
    # defined by center x_T and radius r_D.
    print("start")
    sc_size = 2*r_D + 1
    ssd = np.zeros((sc_size, sc_size))
    template = getWarpedPatch(I_R, getSimWarp(0,0,0,1), x_T, r_T)

    for i in range(sc_size):
        for j in range(sc_size):
            candidate = getWarpedPatch(I, getSimWarp(i - r_D,j - r_D ,0,1), x_T, r_T)
            diff = template - candidate
            ssd[i,j] = np.sum(diff**2)
    
    x = int(np.argmin(ssd) / ssd.shape[1]) - r_D
    y = int(np.argmin(ssd) % ssd.shape[1]) - r_D
    print("end")
    return (np.array([x,y]), ssd)



def conv2x(image):
    sbX = np.array((
        [0, 0, 0],
        [1, 0, -1],
        [0, 0, 0]), dtype="int")
    Ix = convolve2d(image, sbX, mode='valid')
    return Ix

def conv2y(image):
    sbY = np.array((
        [0, 1, 0],
        [0, 0, 0],
        [0,-1, 0]), dtype="int")
    Iy = convolve2d(image, sbY, mode='valid')
    return Iy


def trackKLTRobustly(I_prev, I, keypoint, r_T, num_iters, llambda):
    # I_prev: reference image, I: image to track point in, keypoint: point to 
    # track, expressed as [x y]=[col row], r_T: radius of patch to track, 
    # num_iters: amount of iterations to run, lambda: bidirectional error
    # threshold; delta_keypoint: delta by which the keypoint has moved between 
    # images, (2x1), keep: true if the point tracking has passed the
    # bidirectional error test.
    keep = False
    W, hist = trackKLT(I_prev, I, keypoint, r_T, num_iters)
    delta = W[:, -1]
    Winv, p_hist = trackKLT(I, I_prev, keypoint + delta, r_T, num_iters)
    delta_inv = Winv[: , -1]
    keep = np.linalg.norm(delta + delta_inv) < llambda
    
    return (W, hist, keep)


def trackKLT(I_R, I, x_T, r_T, num_iters):
    # I_R: reference image, I: image to track point in, x_T: point to track,
    # expressed as [x y]=[col row], r_T: radius of patch to track, num_iters:
    # amount of iterations to run; W(2x3): final W estimate, p_hist 
    # (6x(num_iters+1)): history of p estimates, including the initial
    # (identity) estimate
    p_hist = np.zeros((6,num_iters+1))
    W = getSimWarp(0,0,0,1)
    p_hist[:,0] = W.flatten()

    ref_img = getWarpedPatch(I_R, W, x_T, r_T)
    ref_patch = ref_img.flatten()

    xs = np.arange(-r_T, r_T + 1)
    n = 2*r_T + 1
    N = n*n
    x_col = np.concatenate((np.kron(np.ones((1,n)), xs).T, np.kron(xs, np.ones((1,n))).T, np.ones((N,1))), axis = 1)
    dwdx = np.kron(x_col, np.eye(2))

    for iter in range(num_iters):

        # compute warped patch
        wraped_patch_conv = getWarpedPatch(I, W, x_T, r_T + 1)
        wraped_patch_img = wraped_patch_conv[1:-1, 1:-1]
        wraped_patch = wraped_patch_img.flatten()

        # comupute the error 
        error = (ref_patch.astype(np.float) - wraped_patch.astype(np.float)).reshape(N,1)

        # compute the gradients
        wrap_x = conv2x(wraped_patch_conv)
        wrap_y = conv2y(wraped_patch_conv)
        grad_xy = np.vstack((wrap_x.flatten(), wrap_y.flatten())).T

        didp = np.zeros((N , 6))

        for i in range(N):
            didp[i, :] = np.dot( grad_xy[i,:], dwdx[i*2: i*2 + 2, :])

        H = np.dot(didp.T, didp)

        temp = np.dot(didp.T, error)

        delta_p = np.dot( np.linalg.inv(H) , temp)
        W = W + delta_p.reshape(2,3, order ='F')

        p_hist[:, iter + 1] = W.flatten()
    
        if np.linalg.norm(delta_p) < 1e-3:
            p_hist = p_hist[:, 0:iter + 1]
            break

    return ( W , p_hist )



