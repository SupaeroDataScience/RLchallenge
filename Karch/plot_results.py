import numpy as np
import matplotlib.pyplot as plt

comment = '300000_alpha00001'
scores = np.load('results/scores' + comment + '.npy')
final_score = np.reshape(np.load('results/cumulated.npy'), (1, 20))
scores = np.concatenate((scores, final_score))
avg_cost = np.load('results/avg_cost' + comment + '.npy')
max_cost = np.load('results/max_cost' + comment + '.npy')

avglist = []
maxlist = []
minlist = []
for test in scores:
    avglist.append(np.average(test))
    minlist.append(np.min(test))
    maxlist.append(np.max(test))

x_score = 25000 * np.linspace(1, 12, 12)
x_cost = 200 * np.linspace(0, 1500, 1503)
# f, ax = plt.subplots(2, 2)
# ax[0, 0].plot(x_score, avglist)
# ax[1, 0].semilogy(x_cost, avg_cost)
# ax[0, 1].plot(x_score, maxlist)
# ax[0, 1].plot(x_score, minlist)
# ax[1, 1].semilogy(x_cost, max_cost)
# plt.show()

plt.figure()
plt.plot(x_score, avglist, linestyle='-.', marker='o', label='Mean')
plt.plot(x_score, minlist, linestyle='-.', marker='o', label='Min')
plt.plot(x_score, maxlist, linestyle='-.', marker='o', label='Max')
plt.ylabel('Score')
plt.xlabel('Iteration')
plt.grid()
plt.legend()

plt.figure()
plt.errorbar(x_score, avglist, yerr=[minlist,maxlist], fmt='o',barsabove=True,uplims=True,lolims=True,capsize=None)
plt.plot(x_score, avglist, linestyle='-.', marker='o', label='Median')

plt.figure()
plt.semilogy(x_cost, avg_cost, label='Mean cost over 200 iterations')
plt.semilogy(x_cost, max_cost, label='Max cost over 200 iterations  ')
plt.ylabel('Cost')
plt.xlabel('Iteration')
plt.grid()
plt.legend()
