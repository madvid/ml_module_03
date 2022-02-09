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
    # Parsing the arguments:
    parameters = sys.argv
    if len(parameters) == 1:
        usage()
        sys.exit()

    if len(parameters) != 2:
        s = "Only `-zipcode=x` is exptected."
        print(s, file=sys.stderr)
        sys.exit()

    split = parameters[1].split('=')
    if (len(split) != 2) or (split[0] != '-zipcode') or (split[1] not in s_nb):
        s = 'Wrong parameters.'
        print(s, file=sys.stderr)
        sys.exit()
    
    target = int(split[1])
    
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

    # Transformation of y to 0 or 1 depending on target
    y_ = labelbinarizer(y, target)

    # Spliting the data into a training a test set
    x_train, x_test, y_train, y_test = data_spliter(x.values, y_, 0.2)

    # Preprocessing (simple standardistation of the features)
    scaler_x = MyStandardScaler()
    scaler_x.fit(x_train)
    x_train_tr = scaler_x.transform(x_train)
    x_test_tr = scaler_x.transform(x_test)

    # Instanciation and training of the model
    monolr = MyLogR(np.random.rand(x.shape[1] + 1, 1), alpha=1e-2, max_iter=10000)
    monolr.fit_(x_train_tr, y_train)

    # Prediction and binarization of the probabilities
    pred = monolr.predict_(x_test_tr)
    binarize_pred = binarize(pred, threshold=0.5)

    # Calcul of the fraction of correct prediction
    correct_pred = np.sum(binarize_pred == y_test) / y_test.shape[0]
    print(f"Fraction of the corrected prediction on test set (accuracy):{correct_pred:.4f}")

    # Basic classification metrics (accuracy, recall, precision and f1 scores)
    # Not asked by the subject.
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
