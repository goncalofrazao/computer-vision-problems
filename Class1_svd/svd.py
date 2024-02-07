import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

img = sio.loadmat('image_and_base.mat')

B = img['base30'].astype(int)
# I = img['imb'].astype(int)
I = img['im'].astype(int)

def get_Ichapeu(I, BASE):
    B = np.array([I[i] for i in range(0, len(I), len(I) // BASE)]).T
    C = (np.linalg.inv(B.T @ B) @ B.T) @ I
    return B @ C

def get_Ichapeu_svd(I, BASE):
    U, S, V = np.linalg.svd(I)
    C = np.diag(S) @ V
    return U[:,:BASE] @ C[:BASE,:]

def get_Ichapeu_a_cores(I, BASE):
    Ichapeu = np.zeros_like(I)
    for channel in range(3):  # Loop over color channels
        B = I[:, :BASE, channel]
        C = np.linalg.pinv(B)
        Ichapeu[:, :, channel] = B @ C @ I[:, :, channel]
    return Ichapeu

def get_Ichapeu_svd_a_cores(I, BASE):
    Ichapeu = np.zeros_like(I)
    for channel in range(3):  # Loop over color channels
        U, S, Vt = np.linalg.svd(I[:, :, channel], full_matrices=False)
        S[BASE:] = 0
        Ichapeu[:, :, channel] = U @ np.diag(S) @ Vt
    return Ichapeu

# BASE = 512
# Ichapeu = get_Ichapeu_a_cores(I, BASE)
# Ichapeu = get_Ichapeu_svd_a_cores(I, BASE)

# plt.imshow(Ichapeu)
# plt.show()


# loss1 = []
# loss2 = []
# bases = []
# i = 1

# while i <= 1024:
#     bases.append(i)
#     Ichapeu = get_Ichapeu(I, i)
#     loss1.append(np.linalg.norm(I - Ichapeu))
    
#     plt.subplot(1, 2, 1)
#     plt.imshow(Ichapeu, cmap='gray')
    
#     Ichapeu = get_Ichapeu_svd(I, i)
#     loss2.append(np.linalg.norm(I - Ichapeu))
    
#     plt.subplot(1, 2, 2)
#     plt.imshow(Ichapeu, cmap='gray')

#     plt.suptitle('BASE = ' + str(i))
#     plt.show()
#     i *= 2

# plt.plot(bases, loss1, label='loss1')
# plt.plot(bases, loss2, label='loss2')
# plt.legend()
# plt.show()
