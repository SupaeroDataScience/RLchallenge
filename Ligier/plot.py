import numpy as np
import matplotlib.pyplot as plt

# Import data
data = np.loadtxt('eval.log',delimiter=',')

# Plot
plt.figure(figsize=(12,8))
plt.plot(data[:,0], data[:,1], label='mean')
plt.plot(data[:,0], data[:,2], label='max')
plt.legend()
plt.xlabel('# of epochs (10000 steps)')
plt.ylabel('Score')
plt.title('Evolution of the mean and max score during training')

plt.savefig('eval.svg',format='svg')
