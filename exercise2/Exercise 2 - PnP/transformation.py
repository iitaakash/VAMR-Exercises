import numpy as np

def rotMatrix2Quat(M):
    i,j,k = 0,1,2
    if M[1,1] > M[0,0]:
        i,j,k = 1,2,0
    if M[2,2] > M[i,i]:
        i,j,k = 2,0,1
    t = M[i,i] - (M[j,j] + M[k,k]) + 1
    q = np.zeros((4,1))
    q[0] = M[k,j] - M[j,k]
    q[i+1] = t
    q[j+1] = M[i,j] + M[j,i]
    q[k+1] = M[k,i] + M[i,k]
    q = q * 0.5 / np.sqrt(t)
    return q


def quat2RotMatrix(q):
    R = np.array([[q(1)^2 + q(2)^2 - q(3)^2 - q(4)^2,  2*(q(2)*q(3) + q(1)*q(4)),          2*(q(2)*q(4) - q(1)*q(3))], \
                 [2*(q(2)*q(3) - q(1)*q(4)),           q(1)^2 - q(2)^2 + q(3)^2 - q(4)^2,  2*(q(3)*q(4) + q(1)*q(2))], \
                 [2*(q(2)*q(4) + q(1)*q(3)),           2*(q(3)*q(4) - q(1)*q(2)),          q(1)^2 - q(2)^2 - q(3)^2 + q(4)^2]]).T
    return R
