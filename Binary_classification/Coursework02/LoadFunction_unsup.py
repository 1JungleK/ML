import pandas as pd
import numpy as np


def load_data():
    data = pd.read_csv("../Data/Data.csv")

    row_indexes = data[data["Label"].isin([2])].index
    data.drop(row_indexes, inplace=True)

    data = np.array(data)
    x = data[:, 1:16]

    return x
