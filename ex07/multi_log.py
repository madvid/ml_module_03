import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = os.path.join(os.path.dirname(__file__), '..', 'ex06')
sys.path.insert(1, path)
from my_logistic_regression import MyLogisticRegression as MyLogR

path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
from data_spliter import data_spliter

path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
from scaler import MyStandardScaler

# ########################################################################## #
#  _____________________________ CONSTANTES ________________________________ #
# ########################################################################## #
from constants import *

# ########################################################################## #
#  ______________________________ FUNCTIONS ________________________________ #
# ########################################################################## #
from commons import *
# >>> Usage <<<
def usage():
    s = "usgage: python mono_log.py -zipcode=0/1/2/3"
    print(s)

# >>> Preprocessing related methods <<<
def labelbinarizer(y, target):
    y_ = np.zeros(y.shape, dtype='int8')
    y_[np.where(y == target)] = 1
    return y_

def binarize(y, threshold=0.0, copy=True):
    """Cheap mimic of the binarize method from sklearn.preprocessing module
    """
    if copy:
        y_ = np.zeros(y.shape, dtype=np.int8)
        y_[y >= threshold] = 1
        return y_
    y[y >= threshold] = 1
    y[y < threshold] = 0

# ########################################################################## #
#  ________________________________ MAIN ___________________________________ #
# ########################################################################## #

if __name__ == "__main__":    
    # Importing data:
    try:
        x = pd.read_csv(file_x, index_col=0)
        y = pd.read_csv(file_y, index_col=0)
    except:
        s = "Issue while reading one of the dataset."
        print(s, file=sys.stderr)
        sys.exit()
 
    try:
        # casting the y data
        # 2 reasons: minimizing the memory space
        #            if casting fails it means y is not numeric only
        y = y.to_numpy(dtype=np.int8)
    except:
        s = "Something wrong when casting data to integer."
        print(s, file=sys.stderr)
        sys.exit()

    if x.shape[0] != y.shape[0]:
        s = f"Unmatching number of lines between {file_x} and {file_y}"
        print(s, file=sys.stderr)
        sys.exit()

    # Spliting the data into a training a test set
    x_train, x_test, y_train, y_test = data_spliter(x.values, y, 0.2)

    # Preprocessing (simple standardistation of the features)
    scaler_x = MyStandardScaler()
    scaler_x.fit(x_train)
    x_train_tr = scaler_x.transform(x_train)
    x_test_tr = scaler_x.transform(x_test)

    # Instanciation and training of the model
    monolr_Venus = MyLogR(np.random.rand(x.shape[1] + 1, 1), alpha=1e-2, max_iter=10000)
    monolr_Earth = MyLogR(np.random.rand(x.shape[1] + 1, 1), alpha=1e-2, max_iter=10000)
    monolr_Mars = MyLogR(np.random.rand(x.shape[1] + 1, 1), alpha=1e-2, max_iter=10000)
    monolr_AstroBelt = MyLogR(np.random.rand(x.shape[1] + 1, 1), alpha=1e-2, max_iter=10000)
    
    monolr_Venus.fit_(x_train_tr, labelbinarizer(y, 0))
    monolr_Earth.fit_(x_train_tr, labelbinarizer(y, 1))
    monolr_Mars.fit_(x_train_tr, labelbinarizer(y, 2))
    monolr_AstroBelt.fit_(x_train_tr, labelbinarizer(y, 3))

    # Prediction and binarization of the probabilities
    pred_Venus = monolr_Venus.predict_(x_test_tr)
    pred_Earth = monolr_Earth.predict_(x_test_tr)
    pred_Mars = monolr_Mars.predict_(x_test_tr)
    pred_AstroBelt = monolr_AstroBelt.predict_(x_test_tr)
    
    preds = np.hstack((pred_Venus, pred_Earth, pred_Mars, pred_AstroBelt))
    oneVsAll_pred = np.argmax(preds, axis=1)

    # Calcul of the fraction of correct prediction
    correct_pred = np.sum(oneVsAll_pred == y_test) / y_test.shape[0]
    print(f"Fraction of the corrected prediction on test set (accuracy):{correct_pred:.4f}")

    # Basic classification metrics (accuracy, recall, precision and f1 scores)
    # Not asked by the subject.
    sys.exit()
    metrics_report(y_test.astype(np.int8), binarize_pred)

    # Plotting of the data and the predictions
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].scatter(x_test[:,0], y_test, s=24, label='true value')
    axes[0].scatter(x_test[:,0], binarize_pred, s=12, label='prediction')
    axes[1].scatter(x_test[:,1], y_test, s=24, label='true value')
    axes[1].scatter(x_test[:,1], binarize_pred, s=12, label='prediction')
    axes[2].scatter(x_test[:,2], y_test, s=24, label='true value')
    axes[2].scatter(x_test[:,2], binarize_pred, s=12, label='prediction')

    citizen  = list(dct_labels.keys())[target]
    axes[0].set_ylabel("citizen of " + citizen)
    axes[0].set_xlabel(x.columns[0])
    axes[1].set_xlabel(x.columns[1])
    axes[2].set_xlabel(x.columns[2])

    axes[0].legend(), axes[1].legend(), axes[2].legend()
    axes[0].grid(), axes[1].grid(),axes[2].grid()
    plt.show()
