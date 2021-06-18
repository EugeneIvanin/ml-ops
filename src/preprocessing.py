import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    params = yaml.safe_load(open("params.yaml"))["preprocessing"]
    test_share = params["test_share"]
    seed = params["seed"]
    one_hot = params["one_hot"]
    normalize_flag = params["normalize_flag"]

    data = pd.read_csv('data/crx.data', header=None, na_values='?')
    data.columns = ['A' + str(i) for i in range(1, 16)] + ['class']
    categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']
    numerical_columns = [c for c in data.columns if data[c].dtype.name != 'object']
    data = data.fillna(data.median(axis=0), axis=0)
    data['A1'] = data['A1'].fillna('b')
    data_describe = data.describe(include=[object])
    for c in categorical_columns:
        data[c] = data[c].fillna(data_describe[c]['top'])

    binary_columns = [c for c in categorical_columns if data_describe[c]['unique'] == 2]
    nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]
    data.at[data['A1'] == 'b', 'A1'] = 0
    data.at[data['A1'] == 'a', 'A1'] = 1
    data_describe = data.describe(include=[object])
    for c in binary_columns[1:]:
        top = data_describe[c]['top']
        top_items = data[c] == top
        data.loc[top_items, c] = 0
        data.loc[np.logical_not(top_items), c] = 1
    if one_hot == 'True':
        data_nonbinary = pd.get_dummies(data[nonbinary_columns])
    else:
        data_nonbinary = data[nonbinary_columns]

    data_numerical = data[numerical_columns]

    if normalize_flag == 'True':
        data_numerical = (data_numerical - data_numerical.mean()) / data_numerical.std()

    data = pd.concat((data_numerical, data[binary_columns], data_nonbinary), axis=1)
    data = pd.DataFrame(data, dtype=float)
    X = data.drop('class', axis=1)
    y = data['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_share, random_state=seed)
    X_train.to_csv('data/X_train.csv')
    X_test.to_csv('data/X_test.csv')
    y_train.to_csv('data/y_train.csv')
    y_test.to_csv('data/y_test.csv')


if __name__ == "__main__":
    main()
