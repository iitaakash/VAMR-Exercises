import numpy as np 

def Cross2Matrix(x):
    out = np.array([0, -x[2], x[1], x[2], 0, -x[0],-x[1], x[0], 0]).reshape((3,3))
    return out

def LinearTriangulation(p1,p2,m1,m2):
    num_pts = p1.shape[1] 
    all_points = np.zeros((4,num_pts))
    for i in range(num_pts):
        p1cross = np.dot(Cross2Matrix(p1[:,i]), m1)
        p2cross = np.dot(Cross2Matrix(p2[:,i]), m2)
        A = np.concatenate((p1cross, p2cross), axis = 0)
        _, _, Vh = np.linalg.svd(A, full_matrices=True)
        V = Vh.T
        point = V[:,-1] 
        point = point / point[3]
        all_points[:,i] = point
    return all_points


def EstimateEssentialMatrix(p1,p2,K1,K2):
    F = FundamentalEightPoint_Normalized(p1,p2)
    return np.dot(np.dot(K2.transpose(), F) , K1)


def FundamentalEightPoint(p1, p2):
    p1 = p1 / p1[2,:]
    p2 = p2 / p2[2,:]
    num_pts = p1.shape[1]

    Q  = np.zeros((num_pts,9))

    for i in range(num_pts):
        Q[i,:] = np.kron(p2[:,i], p1[:,i])
    
    _, _, Vh = np.linalg.svd(Q, full_matrices=True)
    V = Vh.T
    F = V[:,-1].reshape((3,3))

    H, D, Vh = np.linalg.svd(F, full_matrices=True)
    D = np.diag(D)
    D[2,2] = 0.0

    F = np.dot(np.dot(H,D),Vh)
    return F

def FundamentalEightPoint_Normalized(q1, q2):
    p1 = q1 / q1[2,:]
    p2 = q2 / q2[2,:]
    # print(p1)
    # print(p2)
    # p1[0:2,:] = p1[0:2,:] / p1[2,:]
    # p2[0:2,:] = p2[0:2,:] / p2[2,:]
    mean1 = np.mean(p1[0:2,:], axis = 1)
    mean2 = np.mean(p2[0:2,:], axis = 1)

    var1 = np.mean(np.std(p1[0:2,:], axis = 1))
    var2 = np.mean(np.std(p2[0:2,:], axis = 1))

    sj1 = np.sqrt(2) / var1
    sj2 = np.sqrt(2) / var2

    T1 = np.array([[sj1, 0, -1*sj1*mean1[0]],[0,sj1,-1*sj1*mean1[1]],[0,0,1]])
    T2 = np.array([[sj2, 0, -1*sj2*mean2[0]],[0,sj2,-1*sj2*mean2[1]],[0,0,1]])

    p1_new = np.dot(T1, p1) 
    p2_new = np.dot(T2, p2)

    # p1_new = (np.sqrt(2)*(p1 - mean1))/var1
    # p2_new = (np.sqrt(2)*(p2 - mean2))/var2

    F = FundamentalEightPoint(p1_new, p2_new)

    return np.dot(np.dot(T2.transpose(), F), T1)

def DecomposeEssentialMatrix(E):
    U, S, Vt = np.linalg.svd(E, full_matrices=True)
    u3 = U[:,-1]
    W = np.array([0,-1,0,1,0,0,0,0,1]).reshape((3,3))

    R1 = np.dot(np.dot(U,W),Vt).reshape((3,3))
    R2 = np.dot(np.dot(U,W.transpose()),Vt).reshape((3,3))

    if np.linalg.det(R1) < 0:
        R1 = -1 * R1
    if np.linalg.det(R2) < 0:
        R2 = -1 * R2
    
    R = np.stack((R1,R2), axis = 2)
    return (R, u3.reshape(3,1))


def DisambiguateRelativePose(Rots,u3,p1,p2,K1,K2):
    max_points = 0
    i_final = -1
    r_final = 0
    for i in [-1, 1]:
        for r in [0 , 1]:
            P1 = np.dot(K1, np.eye(3,4))
            P2 = np.dot(K2, np.concatenate((Rots[:,:,r],i*u3.reshape((3,1))), axis = 1))
            points = LinearTriangulation(p1, p2, P1, P2)
            count_pos = np.sum(points[2,:] > 0)
            if count_pos > max_points:
                max_points = count_pos
                i_final = i
                r_final = r
    
    
    return (Rots[:,:,r_final] , i_final * u3)


def DistPoint2EpipolarLine(F, p1, p2):
    num_points = p1.shape[1]

    homog_points = np.concatenate((p1, p2), axis = 1)
    epi_lines = np.concatenate((np.dot(F.transpose(), p2), np.dot(F, p1)), axis = 1)

    denom = epi_lines[0,:] * epi_lines[0,:] + epi_lines[1,:] * epi_lines[1,:]
    sum_val = np.sum(epi_lines * homog_points, axis=0)
    cost = np.sqrt( np.sum((sum_val*sum_val)/denom) / num_points )
    return cost