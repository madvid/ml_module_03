import sys
import os
import pandas as pd
import numpy as np

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

file_x = "solar_system_census.csv"
file_y = "solar_system_census_planets.csv"

dct_labels = {"Venus": 0,
              "Earth": 1,
              "Mars": 2,
              "Asteroids' Belt": 3}

s_nb = ['0', '1', '2', '3']
eps = 1e-8
# ########################################################################## #
#  ______________________________ FUNCTIONS ________________________________ #
# ########################################################################## #

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
        y_ = np.zeros(y.shape)
        y_[y >= threshold] = 1
        return y_
    y[y >= threshold] = 1
    y[y < threshold] = 0

# >>> Scoring related methods <<<
def _check_type_(y, ypred):
    """ Cheecking y and ypred are both numpy ndarray objects
    """
    # Checking type
    if (not isinstance(y, np.ndarray)) \
            or (not isinstance(ypred, np.ndarray)):
        s = "Unexpected type for y or ypred."
        print(s, file=sys.stderr)
        sys.exit()
    # Checking data type
    if y.dtype.kind != ypred.dtype.kind:
        s = "Unmatching data type."
        print(s, file=sys.stderr)
        sys.exit()

def _check_shape_(y, ypred):
    """ Checking y and ypred are both of the same shape
    and are both 1 or 2 dimensional + same dimension (in fact it is redondant).
    """
    # Checking shape along axis 0
    if y.shape != ypred.shape:
        s = "Mismatching shape between y and ypred."
        print(s, file=sys.stderr)
        sys.exit()
    # Checkin data type
    if (y.ndim > 2) \
            or (y.ndim > 2) \
            or (y.ndim != ypred.ndim):
        s = "Unconsistent dimension between y and ypred."
        # Well it should never happens, it would be catch by previous if.
        print(s, file=sys.stderr)
        sys.exit()

def _check_samples_(y, ypred):
    """ Checking the set of values of ypred respectively to y.
    (ie all the value in ypred must appear in y otherwise there is a problem)
    """
    set_y = np.unique(y)
    set_ypred = np.unique(ypred)
    if any([e not in set_y for e in set_ypred]):
        s = "Unexpected value in y_hat."
        print(s, file=sys.stderr)
        sys.exit()

def accuracy_score_(y:np.ndarray, ypred:np.ndarray, pos_label=1):
    """
    Compute the accuracy score.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
    Return:
        The accuracy score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    Reminder:
        accuracy = (TP + TN) / (TP + TF + FP + FN)
        with:
            TP: True Positive
            TN: True Negative
            FP: False Positive
            FN: True Negative
    """
    _check_type_(y, ypred)
    _check_shape_(y, ypred)
    _check_samples_(y, ypred)
    tp_arr = (y == pos_label) & (ypred == pos_label)
    fp_arr = (y != pos_label) & (ypred == pos_label)
    tn_arr = (y != pos_label) & (ypred != pos_label)
    fn_arr = (y == pos_label) & (ypred != pos_label)
    tp = tp_arr.sum()
    fp = fp_arr.sum()
    tn = tn_arr.sum()
    fn = fn_arr.sum()
    if (tp == 0) & (fp == 0) & (tn == 0) & (fn == 0):
        accuracy = 0
    else:
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    return round(accuracy, 4)

def precision_score_(y:np.ndarray, ypred:np.ndarray, pos_label=1):
    """
    Compute the accuracy score.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
    Return:
        The accuracy score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    Reminder:
        accuracy = (TP + TN) / (TP + TF + FP + FN)
        with:
            TP: True Positive
            TN: True Negative
            FP: False Positive
            FN: True Negative
    """
    _check_type_(y, ypred)
    _check_shape_(y, ypred)
    _check_samples_(y, ypred)
    tp_arr = (y == pos_label) & (ypred == pos_label)
    fp_arr = (y != pos_label) & (ypred == pos_label)
    tp = tp_arr.sum()
    fp = fp_arr.sum()
    precision = tp / (tp + fp + eps)
    return round(precision, 4)

def recall_score_(y:np.ndarray, ypred:np.ndarray, pos_label=1):
    """
    Compute the recall score.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
    Return:
        The recall score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    Reminder:
        recall = TP / (TP + FN)
        with:
            TP: True Positive
            TN: True Negative
            FP: False Positive
            FN: True Negative
    """
    _check_type_(y, ypred)
    _check_shape_(y, ypred)
    _check_samples_(y, ypred)
    tp_arr = (y == pos_label) & (ypred == pos_label)
    fn_arr = (y == pos_label) & (ypred != pos_label)
    tp = tp_arr.sum()
    fn = fn_arr.sum()
    recall = tp / (tp + fn + eps)
    return round(recall, 4)

def specificity_score_(y:np.ndarray, ypred:np.ndarray, pos_label=1):
    """
    Compute the specificity score.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
    Return:
        The specificity score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    Reminder:
        specificity = TN / (TN + FP)
        with:
            TP: True Positive
            TN: True Negative
            FP: False Positive
            FN: True Negative
    """
    _check_type_(y, ypred)
    _check_shape_(y, ypred)
    _check_samples_(y, ypred)
    tp_arr = (y == pos_label) & (ypred == pos_label)
    fp_arr = (y != pos_label) & (ypred == pos_label)
    tn_arr = (y != pos_label) & (ypred != pos_label)
    fn_arr = (y == pos_label) & (ypred != pos_label)

    fp = fp_arr.sum()
    tn = tn_arr.sum()
    specificity = tn / (tn + fp + eps)
    return round(specificity, 4)

def f1_score_(y:np.ndarray, ypred:np.ndarray, pos_label=1):
    """
    ...Docstring...
    """
    _check_type_(y, ypred)
    _check_shape_(y, ypred)
    _check_samples_(y, ypred)
    precision = precision_score_(y, ypred, pos_label)
    recall = recall_score_(y, ypred, pos_label)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return round(f1, 4)

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
        print(s, sys.stderr)
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
        y.astype('int8')
    except:
        s = "Something wrong when casting data to integer."
        print(s, file=sys.stderr)
        sys.exit()

    if x.shape[0] != y.shape[0]:
        s = f"Unmatching number of lines between {file_x} and {file_y}"
        print(s, file=sys.stderr)
        sys.exit()
    
    y_ = labelbinarizer(y, target)
    
    x_train, x_test, y_train, y_test = data_spliter(x.values, y_, 0.2)
    
    scaler_x = MyStandardScaler()
    # scaler_y = MyStandardScaler()
    scaler_x.fit(x_train)
    # scaler_y.fit(y_train)
    x_train_tr = scaler_x.transform(x_train)
    x_test_tr = scaler_x.transform(x_test)

    monolr = MyLogR(np.random.rand(x.shape[1] + 1, 1), alpha=1e-2, max_iter=10000)
    monolr.fit_(x_train_tr, y_train)

    pred = monolr.predict_(x_test_tr)
    binarize_pred = binarize(pred, threshold=0.5)

    correct_pred = (binarize_pred == y_test)
    print(correct_pred.sum())