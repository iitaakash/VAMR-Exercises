import matplotlib.pyplot as plt
import numpy as np



## cannot verity the correctness of this function
## NOTE: please dont use this
def plotTrajectory3D(translation ,rotation, pts3d):
    #  -translation(N,3): translations
    #  -rotation(N,3,3): orientations given by rot mat
    #  -pts3d(N,4): additional 3D points to plot

    arrow_len = 0.05

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Generate the values
    x_vals = pts3d[:,0]
    y_vals = pts3d[:,1]
    z_vals = pts3d[:,2]

    # Plot the values
    ax.scatter(x_vals, y_vals, z_vals)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_xlim((-0.20, 0.20))
    ax.set_ylim((-0.20, 0.30))
    ax.set_zlim((-0.30, 0.0))
    azim = 45
    elev = -140
    ax.view_init(elev = elev, azim = azim)

    for i in range(translation.shape[0]):
        rot = rotation[i].T
        trans = -1.0 * np.dot(rot, (translation[i]).reshape((3,1)))
        trans = trans.reshape((3,))

        # plotting pose
        ax_pt = ax.scatter(trans[0],trans[1],trans[2], c="g")
        x_tf = ax.quiver(trans[0],trans[1],trans[2], rot[0,0],rot[0,1],rot[0,2], color='r', length=arrow_len, normalize=True)
        y_tf = ax.quiver(trans[0],trans[1],trans[2], rot[1,0],rot[1,1],rot[1,2], color='g',length=arrow_len, normalize=True)
        z_tf = ax.quiver(trans[0],trans[1],trans[2], rot[2,0],rot[2,1],rot[2,2], color='b',length=arrow_len, normalize=True)
        plt.pause(0.05)
        plt.draw()
        ax_pt.remove()
        x_tf.remove()
        y_tf.remove()
        z_tf.remove()

    plt.show()
