import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

# Generate a sample from a normal distribution
np.random.seed(0)  # for reproducibility
data = np.random.normal(0, 1, 1000)  # mean=0, standard deviation=1

# Create the probability plot
res = stats.probplot(data, plot=plt)
plt.title('Probability Plot - Check for Normality')

# Save the plot as a PNG image
plt.savefig('probability_plot.png')

plt.show()
