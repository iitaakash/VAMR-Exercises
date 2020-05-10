import numpy as np 


# K -> (4,4)
# points3d -> (12,4)
# points2d -> (12, 3)
def GetCameraPosePNP(points3d, points2d, K):
    # calibrate pixels
    Kinv = np.linalg.inv(K)
    cal_pts = np.dot(Kinv[0:3,0:3], points2d.T)
    cal_pts = cal_pts[0:2,:] / cal_pts[2,:]

    # q matrix generate
    temp2d = -1.0 * cal_pts.flatten("F").reshape(-1,1)
    temp3d = np.repeat(points3d, 2, axis = 0)
    col1 = temp3d.copy()
    col2 = temp3d.copy()
    col1[1::2] = 0.0
    col2[::2] = 0.0
    col3 = temp3d * temp2d
    Q = np.hstack((col1, col2, col3))

    # solve eqns
    _, _, Vh = np.linalg.svd(Q, full_matrices=True)
    V = Vh.T
    M = V[:,-1].reshape((3,4))

    # enforce determinent
    if np.linalg.det(M[0:3,0:3]) < 0.0:
        M = -1.0 * M

    # find R
    R = M[:,0:3]
    ur, _, vr = np.linalg.svd(R, full_matrices=True)
    R1 = np.dot(ur,vr)

    # scale estimation
    scale = np.linalg.norm(R1, ord = 'fro') / np.linalg.norm(R, ord = 'fro')

    M[:,0:3] = R1
    M[:,3] = M[:,3]*scale
    return M


def ReprojectPoints(P, M, K):
    p_mat = np.eye(4)
    p_mat[0:3,:] = M
    pts = np.dot(K,np.dot(p_mat,P.T))
    px = pts[0:2,:] / pts[2,:]
    return px.T