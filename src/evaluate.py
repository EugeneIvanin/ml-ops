from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import fbeta_score
import yaml
import pandas as pd
import pickle


def main():
    params = yaml.safe_load(open("params.yaml"))["evaluate"]
    beta = params['beta']
    y_pred_test = pd.read_csv('data/y_pred_test.csv', header=None)[1]
    y_test = pd.read_csv('data/y_test.csv').iloc[:, 1]

    acc = accuracy_score(y_test, y_pred_test)
    roc_auc = roc_auc_score(y_test, y_pred_test)
    fbeta = fbeta_score(y_test, y_pred_test, beta=beta)

    with open("metrics.txt", "w") as metrics:
        metrics.write("Accuracy: {}, ".format(acc))
        metrics.write("Roc-auc: {}, ".format(roc_auc))
        metrics.write("F-beta: {}, ".format(fbeta))


if __name__ == "__main__":
    main()
