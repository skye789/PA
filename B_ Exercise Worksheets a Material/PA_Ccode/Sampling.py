import numpy as np
import matplotlib.pyplot as plt


def gaussian(mu, sigma, X):
    return np.exp(-(X-mu)* (X-mu)/(2*sigma*sigma)) / (sigma*np.sqrt(2*np.pi))

mu = 1
sigma = 0.2
n_samples = 1000
bins = 30
Y_sampled = np.random.normal(mu, sigma, n_samples)
x = np.linspace(0, 2)
Y_truth = gaussian(mu, sigma, x)

plt.subplot(121)
plt.hist(Y_sampled, bins=bins, density=True)##"density=True" is pdf
plt.subplot(122)
plt.plot(x, Y_truth, 'r')
plt.suptitle('Figure 1:Sampled and Ground Truth Distribution in 1D')
plt.show()





