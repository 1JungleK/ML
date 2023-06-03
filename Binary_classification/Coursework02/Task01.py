import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

Data = pd.read_csv("../Data/Data.csv")
features = [i for i in range(1, 16)]

row_indexes = Data[Data["Label"].isin([2])].index
Data.drop(row_indexes, inplace=True)

Data = np.array(Data)
x = Data[:, 1:16]
y = Data[:, 16]
Data_pca = PCA()
Data_pca.fit(x)

print(Data_pca.explained_variance_ratio_.cumsum())

# Results of PCA
plt.plot(features, Data_pca.explained_variance_ratio_.cumsum(), linewidth=1, marker='o', markerfacecolor='white')
plt.ylabel('Cumulative sum of variance')
plt.xlabel('Dimensionality of principle components')
plt.xticks(features)
plt.yticks(np.linspace(0.15, 1, 20, endpoint=True))
plt.grid(alpha=0.5)
plt.title('Explained variance ratio')
plt.show()

# Dimension Reduction
Data_x = PCA(n_components=2)
Data_x.fit(x)
Data_x = Data_x.transform(x)

print(Data_x)

# Dataset Division
ratio = 0.8
training_x = Data_x[:int(ratio * len(Data_x))]
testing_x = Data_x[int(ratio * len(Data_x)):]

train_label = y[:int(ratio * len(y))]
test_label = y[int(ratio * len(y)):]

colors = y
colors_train = train_label
colors_test = test_label

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.scatter(Data_x[:, 0], Data_x[:, 1], c=colors, s=3, cmap="viridis", )
plt.xlabel('feature_01')
plt.ylabel('feature_02')
plt.title('Data set')
plt.colorbar()

# Visualize the data set
plt.subplot(2, 2, 2)
plt.scatter(training_x[:, 0], training_x[:, 1], c=colors_train, s=3, cmap="viridis", )
plt.xlabel('feature_01')
plt.ylabel('feature_02')
plt.title('Training set')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.scatter(testing_x[:, 0], testing_x[:, 1], c=colors_test, s=3, cmap="viridis", )
plt.xlabel('feature_01')
plt.ylabel('feature_02')
plt.title('Testing set')
plt.colorbar()

plt.tight_layout()
plt.show()
