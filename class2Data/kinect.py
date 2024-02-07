import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

calib = sio.loadmat('calib_asus.mat')

depth_data = sio.loadmat('depth_10.mat')
depth_data = depth_data['depth_array']

k_rgb = calib['RGB_cam'][0][0][0]
k_depth = calib['Depth_cam'][0][0][0]
r = calib['R_d_to_rgb']
t = calib['T_d_to_rgb']
r_t = np.concatenate((r, t), axis=1)

def get_uv_rgb(uv_depth, z):
    uv_rgb = k_rgb @ r_t @ np.concatenate((z * np.linalg.inv(k_depth) @ uv_depth, np.array([1])))
    return uv_rgb / uv_rgb[2]


# generate rgbd image
rgb_data = plt.imread('rgb_image_10.png')
print(rgb_data.shape)

rgbd = np.zeros((480, 640, 3))
for i in range(480):
    for j in range(640):
        uv_depth = np.array([i, j, 1])
        z = depth_data[i, j]
        if z == 0:
            continue
        uv_rgb = get_uv_rgb(uv_depth, z)
        rgbd[i, j, :] = rgb_data[int(uv_rgb[0]), int(uv_rgb[1]), :]

plt.imshow(rgbd)
plt.show()
