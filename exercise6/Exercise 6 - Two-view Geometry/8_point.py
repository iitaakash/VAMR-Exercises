import numpy as np
from PoseEstimation import *

N = 40
X = np.random.normal(size=(4,N))

# Simulated scene with error-free correspondences
X[2, :] = X[2, :] * 5 + 10
X[3, :] = 1

P1 = np.array([500, 0, 320, 0,0 ,500 ,240, 0, 0, 0, 1, 0]).reshape((3,4))

P2 = np.array([500, 0, 320, -100, 0, 500, 240, 0, 0, 0, 1, 0]).reshape((3,4))
				
x1 = np.dot(P1, X)     # Image (i.e., projected) points
x2 = np.dot(P2, X)

sigma = 1e-1
noisy_x1 = x1 + sigma * np.random.normal(size = x1.shape)
noisy_x2 = x2 + sigma * np.random.normal(size = x1.shape)

## Fundamental matrix estimation via the 8-point algorithm

# Estimate fundamental matrix
# Call the 8-point algorithm on inputs x1,x2
F = FundamentalEightPoint(x1,x2)

# Check the epipolar constraint x2(i).' * F * x1(i) = 0 for all points i.
cost_algebraic = np.linalg.norm(np.sum(x2*np.dot(F,x1), axis = 0)) / np.sqrt(N)
cost_dist_epi_line = DistPoint2EpipolarLine(F,x1,x2)

print('Noise-free correspondences\n')
# print(F)
print('Algebraic error: {}\n'.format(cost_algebraic))
print('Geometric error: {} px\n\n'.format(cost_dist_epi_line))

# Test with noise:

# Estimate fundamental matrix
# Call the 8-point algorithm on noisy inputs x1,x2
F = FundamentalEightPoint(noisy_x1,noisy_x2)

# Check the epipolar constraint x2(i).' * F * x1(i) = 0 for all points i.
cost_algebraic = np.linalg.norm(np.sum(noisy_x2*np.dot(F,noisy_x1), axis = 0)) / np.sqrt(N)
cost_dist_epi_line = DistPoint2EpipolarLine(F,noisy_x1,noisy_x2)

print('Noisy correspondences (sigma={}), with fundamentalEightPoint\n'.format(sigma))
# print(F)
print('Algebraic error: {}\n'.format(cost_algebraic))
print('Geometric error: {} px\n\n'.format(cost_dist_epi_line))


# Normalized 8-point algorithm
# Call the normalized 8-point algorithm on inputs x1,x2
Fn = FundamentalEightPoint_Normalized(noisy_x1,noisy_x2)

# Check the epipolar constraint x2(i).' * F * x1(i) = 0 for all points i.
cost_algebraic = np.linalg.norm(np.sum(noisy_x2*np.dot(Fn,noisy_x1), axis = 0)) / np.sqrt(N)
cost_dist_epi_line = DistPoint2EpipolarLine(Fn,noisy_x1,noisy_x2)


print('Noisy correspondences (sigma={}), with fundamentalEightPoint_normalized\n'.format(sigma))
# print(F)
print('Algebraic error: {}\n'.format(cost_algebraic))
print('Geometric error: {} px\n\n'.format(cost_dist_epi_line))
