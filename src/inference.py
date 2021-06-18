import pandas as pd
import pickle
from numpy import savetxt


def main():
    loaded_model = pickle.load(open('model/rf.sav', 'rb'))
    X_test = pd.read_csv('data/X_test.csv')
    y_pred_test = loaded_model.predict(X_test)
    savetxt('data/y_pred_test.csv', y_pred_test, delimiter=',')


if __name__ == "__main__":
    main()
