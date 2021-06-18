import yaml
import pandas as pd
from sklearn import ensemble
import pickle


def main():
    params = yaml.safe_load(open("params.yaml"))["train"]
    seed = params['seed']
    n_est = params['n_estimators']
    max_depth = params['max_depth']

    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv')

    rf = ensemble.RandomForestClassifier(n_estimators=n_est, random_state=seed, max_depth=max_depth)
    rf.fit(X_train, y_train)

    pickle.dump(rf, open('model/rf.sav', 'wb'))


if __name__ == "__main__":
    main()
