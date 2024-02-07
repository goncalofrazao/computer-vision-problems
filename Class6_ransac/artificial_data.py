import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

NOISE_VARIANCE = 0.5
OUTLIERS_VARIANCE = 20
A = 1
B = 2
C = 3
SIZE_OF_DATA = 200
OUTLIERS_PERCENTAGE = 0.5
NUMBER_OF_INLIERS = int(SIZE_OF_DATA * (1 - OUTLIERS_PERCENTAGE))

X,Y = np.meshgrid(np.arange(0, 10, 0.1), np.arange(0, 10, 0.1))

# create data points
data = np.array([X.flatten(), Y.flatten(), np.ones(X.shape[0] * X.shape[1])]).T

# define plane
plane = np.array([A, B, C])

# calculate Z
Z = plane @ data.T + np.random.normal(0, NOISE_VARIANCE, data.shape[0])

# add Z to data
data = np.concatenate((data[:, :2], Z.reshape(-1, 1)), axis=1)

# shuffle data
np.random.shuffle(data)

# cat data
inliers = data[:NUMBER_OF_INLIERS]
outliers = data[NUMBER_OF_INLIERS:SIZE_OF_DATA]
outliers[:, 2] += np.random.normal(0, OUTLIERS_VARIANCE, outliers.shape[0])
# outliers[:50, 2] += 20
# outliers[50:100, 2] -= 20

# 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(data[:, 0], data[:, 1], data[:, 2])
ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2])
ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2])
plt.show()

# save data
save_data = np.concatenate((inliers, outliers), axis=0)

# generate .mat file
sio.savemat('artifial_points.mat', {'xyz': save_data})
