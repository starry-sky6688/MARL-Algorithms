import numpy as np
import matplotlib.pyplot as plt

r_vdn = np.load('./model/vdn/7-18/episodes_rewards.npy')
r_qmix = np.load('./model/qmix/7-18/episodes_rewards.npy')
r_1, r_2 = [], []
a, b = 0, 0
for i in range(len(r_vdn)):
    a += r_vdn[i]
    b += r_qmix[i]
    if i > 0 and i % 10 == 0:
        r_1.append(a / 10)
        r_2.append(b / 10)
        a = 0
        b = 0
plt.figure()
plt.plot(range(len(r_1)), r_1, c='r', label='vdn')
plt.plot(range(len(r_2)), r_2, c='b', label='qmix')
plt.legend()
plt.show()
