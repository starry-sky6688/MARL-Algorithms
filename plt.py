import numpy as np
import matplotlib.pyplot as plt

r_vdn = np.load('./model/qmix/8m/win_rates.npy')
r_qmix = np.load('./model/qmix/8m/win_rates.npy')
r_1, r_2 = [], []
a, b = 0, 0
num = 1
for i in range(2500):
    a += r_vdn[i]
    b += r_qmix[i]
    if i > 0 and i % num == 0:
        r_1.append(a / num)
        r_2.append(b / num)
        a = 0
        b = 0
plt.figure()
plt.ylim(0, 1.0)
plt.plot(range(len(r_1)), r_1, c='#1E90FF', label='vdn')
plt.plot(range(len(r_2)), r_2, c='#FFA500', label='qmix')
plt.legend()
plt.xlabel('epoch * ' + str(num))
plt.ylabel('win_rate')
plt.show()