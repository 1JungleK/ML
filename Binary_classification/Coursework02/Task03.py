from sklearn.cluster import KMeans
from LoadFunction_unsup import load_data
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

data = load_data()
# n_clusters = list(range(2, 9, 1))
#
# Silhouette_coefficient = []

# for i in n_clusters:
#     kmeans = KMeans(n_clusters=i)
#     labels = kmeans.fit_predict(data)
#     s = silhouette_score(data, labels)
#     Silhouette_coefficient.append(s)
#
# plt.plot(n_clusters, Silhouette_coefficient, linewidth=1, marker='o', markerfacecolor='white')
# plt.ylabel('Silhouette coefficient')
# plt.xlabel('K clusters')
# plt.xticks(n_clusters)
# plt.grid(alpha=0.5)
# plt.show()

Data_x = PCA(n_components=2)
Data_x.fit(data)
Data_x = Data_x.transform(data)


kmeans = KMeans(n_clusters=2)
labels = kmeans.fit_predict(Data_x)

plt.scatter(Data_x[:, 0], Data_x[:, 1], c=labels, s=3, cmap="viridis")
plt.xlabel('feature_1')
plt.ylabel('feature_2')
plt.title('K-means')
plt.show()

# SSE
# SSE = []
# for i in n_clusters:
#
#     kmeans = KMeans(n_clusters=i, random_state=0, n_init="auto")
#     kmeans.fit(data)
#     SSE.append(sum(np.min(cdist(data, kmeans.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])
#
# plt.plot(n_clusters, SSE, 'bx-')
# plt.xlabel('Values of K')
# plt.ylabel('Average Dispersion')
# plt.title('Selecting k with the Elbow Method')
# plt.grid(alpha=0.5)
# plt.show()

# plt.scatter(Data_x[:, 0], Data_x[:, 1], c=labels, s=3, cmap="viridis")
# plt.xlabel('feature_1')
# plt.ylabel('feature_2')
# plt.title('K-means')
# plt.show()
#
#     s = silhouette_score(Data_x, labels)
#     scores.append(s)
#     print(s)
#
# plt.plot(n_clusters, scores, 'bo-')
# plt.xlabel('clusters')
# plt.ylabel('scores')
# plt.title('Testing clusters')
# plt.show()
