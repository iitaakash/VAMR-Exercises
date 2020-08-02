import numpy as np 


def selectKeypoints(scores, num, r):
    # Selects the num best scores as keypoints and performs non-maximum 
    # supression of a (2r + 1)*(2r + 1) box around the current maximum.
    temp_scores = np.pad(scores, r ,'constant')

    keypoints  = np.zeros((num,2)).astype(int)

    for i in range(num):

        # get max index
        kc = np.unravel_index(temp_scores.argmax(), temp_scores.shape)

        # get index in image coord
        x = np.array([kc[0], kc[1]]) - r
        
        # add it to keypoints
        keypoints[i,:] = x.astype(int)

        # set patch to zer0
        temp_scores[kc[0]-r : kc[0]+r+1, kc[1]-r : kc[1]+r+1] = 0.0

    return keypoints
