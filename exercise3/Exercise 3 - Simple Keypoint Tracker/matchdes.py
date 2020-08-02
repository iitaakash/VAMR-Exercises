import numpy as np
from scipy.spatial import distance

def matchDescriptors(query_descriptors, database_descriptors, lmda):

    dist_mat = distance.cdist(query_descriptors, database_descriptors, 'euclidean')
    sorted_dists = np.sort(dist_mat[dist_mat != 0])
    min_dist = sorted_dists[0]

    matches = list(np.ones((query_descriptors.shape[0],)).astype(np.int) * -1)
    matches_dist = list(np.ones((query_descriptors.shape[0],)) * float("inf"))

    dist_mat[dist_mat >= lmda * min_dist] = float("inf")

    min_data_index = np.argmin(dist_mat, axis = 0)
    min_data = np.min(dist_mat, axis = 0)

    for ind_d, ind_q in enumerate(min_data_index):
        if matches[ind_q] == -1:
            matches[ind_q] = ind_d
            matches_dist[ind_q] = min_data[ind_d]
        else:
            # someone has already put a value there
            if min_data[ind_d] < matches_dist[ind_q]:
                matches[ind_q] = ind_d
                matches_dist[ind_q] = min_data[ind_d]

    return matches

