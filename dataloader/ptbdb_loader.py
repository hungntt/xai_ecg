import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def dataloader():
    MI = pd.read_csv('data/ptbdb_abnormal.csv')
    HC = pd.read_csv('data/ptbdb_normal.csv')

    MI.columns = list(range(len(MI.columns)))
    HC.columns = list(range(len(HC.columns)))

    MI = MI.rename({len(MI.columns) - 1: 'Label'}, axis=1)
    HC = HC.rename({len(HC.columns) - 1: 'Label'}, axis=1)

    data = pd.concat([MI, HC], axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)

    y = data['Label'].copy()
    X = data.drop('Label', axis=1).copy()

    return X, y


def example_loader(X, y):
    examples = []
    for i in range(10):
        X_pd = pd.DataFrame({"x": range(0, len(X.iloc[i])), "y": X.iloc[i]})
        examples.append([i, X_pd, "MI" if y.iloc[i] else "HC"])
    return examples
