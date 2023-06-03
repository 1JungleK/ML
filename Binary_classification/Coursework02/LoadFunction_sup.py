from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


def load_data(dimension):
    data = pd.read_csv("../Data/Data.csv")

    row_indexes = data[data["Label"].isin([2])].index
    data.drop(row_indexes, inplace=True)

    data = np.array(data)
    x = data[:, 1:16]
    y = data[:, 16]

    data_pca = PCA(n_components=dimension)
    data_pca.fit(x)

    x = data_pca.transform(x)
    # Dataset Division
    ratio = 0.8
    training_x = x[:int(ratio * len(data))]
    testing_x = x[int(ratio * len(data)):]

    train_label = y[:int(ratio * len(y))]
    test_label = y[int(ratio * len(y)):]

    return training_x, testing_x, train_label, test_label
