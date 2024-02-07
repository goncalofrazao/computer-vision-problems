import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import map_coordinates
im1 = plt.imread('parede1.jpg')
im2 = plt.imread('parede2.jpg')


# Assuming im1 and im2 are numpy arrays
plt.figure()
plt.imshow(im1)
print("Please click")
points1 = np.array(plt.ginput(4))
plt.close()

plt.figure()
plt.imshow(im2)
print("Please click")
points2 = np.array(plt.ginput(4))
plt.close()
a = np.array([]).reshape(0, 9)
for p1, p2 in zip(points1, points2):
    a_prime = np.array([[-p1[0], -p1[1], -1, 0, 0, 0, p2[0]*p1[0], p2[0]*p1[1], p2[0]],
                        [0, 0, 0, -p1[0], -p1[1], -1, p2[1]*p1[0], p2[1]*p1[1], p2[1]]])
    a = np.append(a, a_prime, axis=0)

eigenvalues, eigenvectors = np.linalg.eig(a.T @ a)
solution = eigenvectors[:, np.argmin(eigenvalues)]

homo = solution.reshape(3, 3)
homo = homo / homo[2][2]
print("solution")
print(homo)

# h, status = cv2.findHomography(points1, points2)
# print(h)

# Assuming `im1` is your source image and `homo` is the homography matrix
im_out = cv2.warpPerspective(im1, homo, (im1.shape[1], im1.shape[0]))
plt.imshow(im_out)
plt.show()
