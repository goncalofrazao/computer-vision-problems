import numpy as np
import scipy.io as sio

# load data
# data = sio.loadmat('CalibrationScene_0000.mat')
# print(data.keys())
# print(data['lego'])
P1 = np.random.rand(3, 4)
P2 = np.random.rand(3, 4)

def krt(P):
    Q, R = np.linalg.qr(np.linalg.inv(P[:, :3]))
    K, R = np.linalg.inv(R), Q.T
    tprime = np.linalg.svd(P)[2][-1, :3] / np.linalg.svd(P)[2][-1, -1]
    t = -R @ tprime
    return K, R, t

K1, R1, t1 = krt(P1)
K2, R2, t2 = krt(P2)

R = R2 @ R1.T
t = t2 - R @ t1

uv = np.random.rand(3)
uv = uv / uv[-1]
xyz = np.linalg.inv(K2) @ uv
xyz = R @ xyz + t
uv_hat = K1 @ xyz
uv_hat = uv_hat / uv_hat[-1]
print(uv, uv_hat)
