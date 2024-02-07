import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import random

data = sio.loadmat('planeransac_students.mat')
# data = sio.loadmat('artifial_points.mat')

print(data.keys())

# plot image from data['im]
# plt.imshow(data['im'], cmap='gray')
# plt.show()

# get point cloud
data_points = data['xyz']
print(data_points.shape)

# separate in [x y 1] and [z]
z_points = data_points[:, 2]
xy_points = data_points[:, :2]
xy_points = np.concatenate((xy_points, np.ones((data_points.shape[0], 1))), axis=1)

# shuffle samples
indexes = [i for i in range(data_points.shape[0])]
random.shuffle(indexes)

# define noise and best inliers
noise = 1.5
best_inliers = []
best_zliers = []

for i in range(35):
    # choose 3 random points
    points = xy_points[indexes[i * 3: i * 3 + 3]]
    z = z_points[indexes[i * 3: i * 3 + 3]]

    # calculate the plane equation
    sol = np.linalg.pinv(points) @ z.T

    # calculate the inliers
    condition = np.abs(sol @ xy_points.T - z_points) <= noise
    inliers = xy_points[condition]
    zliers = z_points[condition]
    print(inliers.shape)
    if len(inliers) >= len(best_inliers):
        best_inliers = inliers
        best_zliers = zliers

print(best_inliers.shape)

# calculate the plane equation
sol = np.linalg.pinv(best_inliers) @ best_zliers.T
print(sol.round(2))

# print(variance / samples)