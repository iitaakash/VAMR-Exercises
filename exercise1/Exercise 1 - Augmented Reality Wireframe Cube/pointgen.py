import numpy as np 


# pts returned as 4xN 
def GetChessBoardPts(size = (6,9), res = 0.04):
    x_pts ,y_pts = np.meshgrid(np.arange(size[0]),np.arange(size[1]))
    x_pts = res * x_pts.flatten()
    y_pts = res * y_pts.flatten()
    pts = np.vstack((y_pts, x_pts, np.zeros(x_pts.shape) ,np.ones(x_pts.shape)))
    return pts


# pts as 4xN
def GetCubePoints(pos = np.array([0.04,0.04,0.0]), size = 0.04):
    pts = []
    verts =  np.array([[0,0,0],  #0
                [0,0,1],  #1
                [0,1,0],  #2
                [0,1,1],  #3
                [1,0,0],  #4
                [1,0,1],  #5
                [1,1,0],  #6
                [1,1,1]])  #7
    
    edges = [(0,1),\
             (0,2),\
             (0,4),\
             (7,6),\
             (7,5),\
             (7,3),\
             (2,3),\
             (2,6),\
             (3,1),\
             (4,6),\
             (4,5),\
             (5,1)]

    ind = np.transpose(verts * np.array([1.0,1.0,-1.0])*size)
    pts = np.ones((4, ind.shape[1]))
    pts[0:3,:] = pos.reshape(3,1) + ind

    return pts,edges