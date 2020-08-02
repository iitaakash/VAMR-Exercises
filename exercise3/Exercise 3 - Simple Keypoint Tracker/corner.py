import numpy as np 
from scipy.signal import convolve2d
import cv2



def shi_tomasi(image, patch_size = 9):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sum_Ix2,sum_Iy2,sum_Ixy = GetStructureMatrix(image,patch_size)

    trace, det = GetTrDet(sum_Ix2,sum_Iy2,sum_Ixy)

    shi_tomasi = (trace / 2.0) - (np.sqrt(trace*trace - 4*det) / 2.0)

    shi_tomasi[shi_tomasi < 0.0] = 0.0

    pr = int(np.floor(patch_size / 2.0))
    shi_tomasi = np.pad(shi_tomasi, pr+1 ,'constant')

    return shi_tomasi

def harris(image, patch_size = 9.0, kappa = 0.08):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sum_Ix2,sum_Iy2,sum_Ixy = GetStructureMatrix(image,patch_size)

    trace, det = GetTrDet(sum_Ix2,sum_Iy2,sum_Ixy)

    harris = det - trace * trace * kappa
    
    harris[harris < 0.0] = 0.0

    pr = int(np.floor(patch_size / 2.0))
    harris = np.pad(harris, pr+1 ,'constant')

    return harris


def GetTrDet(sum_Ix2,sum_Iy2,sum_Ixy):
    det = (sum_Ix2 * sum_Iy2) - (sum_Ixy * sum_Ixy)
    trace = sum_Ix2 + sum_Iy2
    return trace, det


def GetStructureMatrix(image,patch_size):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image.astype(np.float)
    
    # construct the Sobel x-axis kernel
    sbX = np.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]), dtype="int")

    # construct the Sobel y-axis kernel
    sbY = np.array((
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]), dtype="int")

    ## conv
    Ix = convolve2d(image, sbX, mode='valid')
    Iy = convolve2d(image, sbY, mode='valid')
    Ixy = Ix*Iy
    Ix2 = Ix*Ix
    Iy2 = Iy*Iy


    sum_ker = np.ones((patch_size,patch_size))

    sum_Ix2 = convolve2d(Ix2, sum_ker, mode = 'valid')
    sum_Iy2 = convolve2d(Iy2, sum_ker, mode = 'valid')
    sum_Ixy = convolve2d(Ixy, sum_ker, mode = 'valid')

    return (sum_Ix2,sum_Iy2,sum_Ixy)