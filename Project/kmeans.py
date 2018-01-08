import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('van/SS_36/cropped.csv', encoding='utf-8')
df.head(3)


# Shape of Datafram
print(df.shape)
# Dataframe headers
print(df.columns.values)


X = df.loc[:, 'Z':'Return Number']

print(X[:3])


K = range(1, 10)
meandistortions = []
for k in K:
    print(k)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')
plt.show()
