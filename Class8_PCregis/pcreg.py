import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

procrustes = sio.loadmat('procrustes_data.mat')
calib = sio.loadmat('calib_asus.mat')

p1 = procrustes['P1']
p2 = procrustes['P2']

# centroid
centroid_p1 = np.mean(p1, axis=0)
centroid_p2 = np.mean(p2, axis=0)

# center the points
centered_p1 = p1 - centroid_p1
centered_p2 = p2 - centroid_p2

A = np.dot(centered_p1.T, centered_p2)
U, D, V = np.linalg.svd(A)
eye = np.eye(3)
eye[2, 2] = np.linalg.det(np.dot(V.T, U.T))
R = V.T @ eye @ U.T

# translation
t = centroid_p2 - np.dot(R, centroid_p1)

# apply transformation
p1_transformed = np.dot(R, p1.T).T + t

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(p1[:, 0], p1[:, 1], p1[:, 2], c='r', marker='o')
ax.scatter(p2[:, 0], p2[:, 1], p2[:, 2], c='b', marker='^')
ax.scatter(p1_transformed[:, 0], p1_transformed[:, 1], p1_transformed[:, 2], c='g', marker='s')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
