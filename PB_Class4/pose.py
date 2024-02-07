import scipy.io as sio
import numpy as np

points = sio.loadmat('calib_asus.mat')

print(points.keys())

# a = np.array([]).reshape(0, 9)

# for u_prime, v_prime, u, v in zip(points['u1'], points['v1'], points['u3'], points['v3']):
#     a_prime = np.array([[u[0], v[0], 1, 0, 0, 0, u_prime[0], v_prime[0], 1],
#                         [0, 0, 0, u[0], v[0], 1, u_prime[0], v_prime[0], 1]])
#     a = np.append(a, a_prime, axis=0)
    
# print(a.shape)
# # Compute the eigenvalues and right eigenvectors of m.T * m
# eigenvalues, eigenvectors = np.linalg.eig(np.dot(a.T, a))

# # The solution is the eigenvector corresponding to the smallest eigenvalue
# solution = eigenvectors[:, np.argmin(eigenvalues)]

# # print(solution)

# # print(a @ solution.reshape(-1, 1))

# P = solution.reshape(3, 3)
# print(P)
