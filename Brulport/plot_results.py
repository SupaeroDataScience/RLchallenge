import pickle
import matplotlib.pyplot as plt

filehander = open("data/results.pickle", "rb")
mean_losses, max_losses, average_scores, max_scores = pickle.load(filehander)
filehander.close()

iterations = [100*(i+1) for i in range(len(mean_losses))]
games = [1000*(i+1) for i in range(len(average_scores))]


plt.figure()
plt.plot(iterations, mean_losses)

plt.figure()
plt.plot(iterations, max_losses)

plt.figure()
plt.plot(games, average_scores)

plt.figure()
plt.plot(games, max_scores)

plt.show()
