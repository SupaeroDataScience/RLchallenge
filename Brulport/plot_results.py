import pickle
import matplotlib.pyplot as plt
import numpy as np

# filehander = open("data/results.pickle", "rb")
# mean_losses, max_losses, average_scores, max_scores = pickle.load(filehander)
# filehander.close()
data = np.load("data/losses.npy")
mean_losses = data[0]
max_losses = data[1]
average_scores = data[2]
max_scores = data[3]

iterations = [5000 + 100 * (i + 1) for i in range(len(mean_losses))]
games = [25000 * (i + 1) for i in range(len(average_scores))]

plt.figure()
ax = plt.subplot(2, 2, 1)
ax.semilogy(iterations, mean_losses)

ax = plt.subplot(2, 2, 2)
ax.semilogy(iterations, max_losses)

ax = plt.subplot(2, 2, 3)
ax.plot(games, average_scores)

ax = plt.subplot(2, 2, 4)
ax.plot(games, max_scores)

plt.show()
