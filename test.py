import numpy as np
import matplotlib.pyplot as plt
import time

# Generate random samples
time_now = int(int(time.strftime("%Y%m%d%H%M%S", time.localtime()))/1e5)
np.random.seed(time_now)  # For reproducibility
l_values = np.random.choice(np.arange(60, 305, 5), 10000000)
r_values = np.random.choice(np.arange(2, 10.2, 0.2), 10000000)

# Compute w
w_values = l_values / r_values
print(np.mean(w_values), np.std(w_values))
# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(w_values, bins=50, edgecolor='k', alpha=0.7)
plt.xlabel('w')
plt.ylabel('Frequency')
plt.title('Distribution of w = l/r')
plt.grid(True)
plt.savefig('temp.png')
plt.show()
