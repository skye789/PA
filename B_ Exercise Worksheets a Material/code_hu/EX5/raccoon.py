import numpy as np
from scipy import misc
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

max_k = 8
num_ref = 10

face = misc.face(gray=True)
len_x = face.shape[1]
len_y = face.shape[0]

# plot Raccoon
fig = plt.figure(figsize=(10, 10))
plt.subplot(221)
plt.imshow(face, cmap = 'gray')
plt.title("Raccoon")

samples = 100000
num_pix = len_x * len_y
pix = np.arange(0, num_pix, 1)

# reshape from 768 * 1024 to 1 * 786432
sface = np.reshape(face, (1, num_pix))
cum_prob = np.cumsum(sface)

# sampling
pix_pos = np.zeros(num_pix)
# Inverse Transform Sampling, take random numbers
R = np.random.randint(0, cum_prob[-1], samples)
# fouund the corresponding position of the pixels
pos = np.interp(R, cum_prob, pix)
np.put(pix_pos, pos.astype(int), 1)
# sample the orignal image using the matrix
sample_point = np.multiply(pix_pos,sface)
# reshape it to reconstruct the image
sample_img = np.reshape(sample_point, (len_y, len_x))
plt.imshow(sample_img, cmap='gray')

X = face.reshape((-1, 1))
pos = pos.astype(int)
X = X[pos]
pos = pos.astype(int)
pos_x = np.mod(pos, 1024)
pos_y = np.round(pos / 1024)
coord = np.column_stack((pos_x, pos_y))
feat = np.column_stack((coord, X))

raccoon_kmeans = KMeans(n_clusters=20).fit(feat)
labels = raccoon_kmeans.labels_

plt.ylim(768, 0)
plt.xlim(0, 1024)

plt.scatter(feat[:, 0], feat[:, 1], c=labels, marker='o', s=(10./fig.dpi)**2)
plt.title("K-means with gap statistics")
plt.show()
