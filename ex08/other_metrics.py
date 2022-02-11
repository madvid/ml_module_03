import sys
import numpy as np
from sklearn.metrics import accuracy_score, \
                            precision_score, \
                            recall_score, \
                            f1_score

# ########################################################################## #
#  _____________________________ CONSTANTES ________________________________ #
# ########################################################################## #
eps = 1e-6

# ########################################################################## #
#  ______________________________ FUNCTIONS ________________________________ #
# ########################################################################## #


def check_type(y, y_hat):
    """
    """
    # Checking type
    if (not isinstance(y, np.ndarray)) \
            or (not isinstance(y_hat, np.ndarray)):
        s = "Unexpected type for y or y_hat."
        print(s, file=sys.stderr)
        sys.exit()
    # Checking data type
    if y.dtype.kind != y_hat.dtype.kind:
        s = "Unmatching data type."
        print(s, file=sys.stderr)
        sys.exit()


def check_shape(y, y_hat):
    """
    """
    # Checking shape along axis 0
    if y.shape[0] != y_hat.shape[0]:
        s = "Unconsistent length between y and y_hat."
        print(s, file=sys.stderr)
        sys.exit()
    # Checkin data type
    if (y.ndim > 2) \
            or (y.ndim > 2) \
            or (y.ndim != y_hat.ndim):
        s = "Unconsistent dimension between y and y_hat."
        print(s, file=sys.stderr)
        sys.exit()


def check_samples(y, y_hat):
    """
    """
    set_y = np.unique(y)
    set_y_hat = np.unique(y_hat)

    if any([e not in set_y for e in set_y_hat]):
        s = "Unexpected value in y_hat."
        print(s, file=sys.stderr)
        sys.exit()


def labelencode(y, y_hat):
    """
    """
    u, y_encoded = np.unique(y, return_inverse=True)
    masks = [np.where(y_hat == label) for label in u]
    y_hat_encoded = np.zeros(y_encoded.shape)
    for ii, mask in enumerate(masks):
        y_hat_encoded[mask] = ii

    return y_encoded, y_hat_encoded


def accuracy_score_(y: np.ndarray, yhat: np.ndarray, pos_label=1):
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
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        with:
            TP: True Positive
            TN: True Negative
            FP: False Positive
            FN: True Negative
    """
    if yhat.ndim > 2:
        str_err = "Incorrect dimension for yhat."
        raise Exception(str_err)
    if y.ndim > 2:
        str_err = "Incorrect dimension for y."
        raise Exception(str_err)
    if yhat.shape != y.shape:
        str_err = "Mismatching shape between yhat and y."
        raise Exception(str_err)
    tp_arr = (y == pos_label) & (yhat == pos_label)
    fp_arr = (y != pos_label) & (yhat == pos_label)
    tn_arr = (y != pos_label) & (yhat != pos_label)
    fn_arr = (y == pos_label) & (yhat != pos_label)
    tp = tp_arr.sum()
    fp = fp_arr.sum()
    tn = tn_arr.sum()
    fn = fn_arr.sum()
    if (tp == 0) & (fp == 0) & (tn == 0) & (fn == 0):
        accuracy = 0
    else:
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    return accuracy


def precision_score_(y: np.ndarray, yhat: np.ndarray, pos_label=1):
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
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        with:
            TP: True Positive
            TN: True Negative
            FP: False Positive
            FN: True Negative
    """
    if yhat.ndim > 2:
        str_err = "Incorrect dimension for yhat."
        raise Exception(str_err)
    if y.ndim > 2:
        str_err = "Incorrect dimension for y."
        raise Exception(str_err)
    if yhat.shape != y.shape:
        str_err = "Mismatching shape between yhat and y."
        raise Exception(str_err)
    if pos_label not in np.unique(y):
        str_err = f"{pos_label} is not a possible value of y"
        raise Exception(str_err)
    tp_arr = (y == pos_label) & (yhat == pos_label)
    fp_arr = (y != pos_label) & (yhat == pos_label)
    tp = tp_arr.sum()
    fp = fp_arr.sum()
    precision = tp / (tp + fp + eps)
    return precision


def recall_score_(y: np.ndarray, yhat: np.ndarray, pos_label=1):
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
            FN: False Negative
    """
    if yhat.ndim > 2:
        str_err = "Incorrect dimension for yhat."
        raise Exception(str_err)
    if y.ndim > 2:
        str_err = "Incorrect dimension for y."
        raise Exception(str_err)
    if yhat.shape != y.shape:
        str_err = "Mismatching shape between yhat and y."
        raise Exception(str_err)
    if pos_label not in np.unique(y):
        str_err = f"{pos_label} is not a possible value of y"
        raise Exception(str_err)
    tp_arr = (y == pos_label) & (yhat == pos_label)
    fn_arr = (y == pos_label) & (yhat != pos_label)
    tp = tp_arr.sum()
    fn = fn_arr.sum()
    recall = tp / (tp + fn + eps)
    return recall


def specificity_score_(y: np.ndarray, yhat: np.ndarray, pos_label=1):
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
    if yhat.ndim > 2:
        str_err = "Incorrect dimension for yhat."
        raise Exception(str_err)
    if y.ndim > 2:
        str_err = "Incorrect dimension for y."
        raise Exception(str_err)
    if yhat.shape != y.shape:
        str_err = "Mismatching shape between yhat and y."
        raise Exception(str_err)
    tp_arr = (y == pos_label) & (yhat == pos_label)
    fp_arr = (y != pos_label) & (yhat == pos_label)
    tn_arr = (y != pos_label) & (yhat != pos_label)
    fn_arr = (y == pos_label) & (yhat != pos_label)

    fp = fp_arr.sum()
    tn = tn_arr.sum()
    specificity = tn / (tn + fp + eps)
    return specificity


def f1_score_(y: np.ndarray, yhat: np.ndarray, pos_label=1):
    """
    Compute the f1 score.
    Args:
        y:a [numpy.array] for the correct labels
        y_hat: [numpy.array] for the predicted labels
        pos_label: [str, int], class on which f1_score is reported (default=1)
    Return:
        The f1 score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if yhat.ndim > 2:
        str_err = "Incorrect dimension for yhat."
        raise Exception(str_err)
    if y.ndim > 2:
        str_err = "Incorrect dimension for y."
        raise Exception(str_err)
    if yhat.shape != y.shape:
        str_err = "Mismatching shape between yhat and y."
        raise Exception(str_err)
    precision = precision_score_(y, yhat, pos_label)
    recall = recall_score_(y, yhat, pos_label)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1

# ########################################################################## #
#  ________________________________ MAIN ___________________________________ #
# ########################################################################## #


if __name__ == "__main__":
    print("# Example 1:")
    y_hat = np.array([[1], [1], [0], [1], [0], [0], [1], [1]])
    y = np.array([[1], [0], [0], [1], [0], [1], [0], [0]])
    # Accuracy
    # your implementation
    res = accuracy_score_(y, y_hat)
    # sklearn implementation
    expected = accuracy_score(y, y_hat)
    print('my accuracy:'.ljust(25), res)
    print('expected accuracy:'.ljust(25), expected)

    # Precision
    # your implementation
    res = precision_score_(y, y_hat)
    # sklearn implementation
    expected = precision_score(y, y_hat)
    print('my precision:'.ljust(25), res)
    print('expected precision:'.ljust(25), expected)

    # Recall
    # your implementation
    res = recall_score_(y, y_hat)
    # sklearn implementation
    expected = recall_score(y, y_hat)
    print('my recall:'.ljust(25), res)
    print('expected recall:'.ljust(25), expected)

    # F1-score
    # your implementation
    res = f1_score_(y, y_hat)
    # sklearn implementation
    expected = f1_score(y, y_hat)
    print('my f1-score:'.ljust(25), res)
    print('expected f1-score:'.ljust(25), expected)

    print("# Example 2:")
    y_hat = np.array(['norminet',
                      'dog',
                      'norminet',
                      'norminet',
                      'dog',
                      'dog',
                      'dog',
                      'dog'])
    y = np.array(['dog',
                  'dog',
                  'norminet',
                  'norminet',
                  'dog',
                  'norminet',
                  'dog',
                  'norminet'])
    # Accuracy
    # your implementation
    # res = accuracy_score_(y, y_hat)
    # #sklearn implementation
    # expected = accuracy_score(y, y_hat)
    # print('my accuracy:'.ljust(25), res)
    # print('expected accuracy:'.ljust(25), expected)

    # Precision
    # your implementation
    res = precision_score_(y, y_hat, pos_label='dog')
    # sklearn implementation
    expected = precision_score(y, y_hat, pos_label='dog')
    print('my precision:'.ljust(25), res)
    print('expected precision:'.ljust(25), expected)

    # Recall
    # your implementation
    res = recall_score_(y, y_hat, pos_label='dog')
    # sklearn implementation
    expected = recall_score(y, y_hat, pos_label='dog')
    print('my recall:'.ljust(25), res)
    print('expected recall:'.ljust(25), expected)

    # F1-score
    # your implementation
    res = f1_score_(y, y_hat, pos_label='dog')
    # sklearn implementation
    expected = f1_score(y, y_hat, pos_label='dog')
    print('my f1-score:'.ljust(25), res)
    print('expected f1-score:'.ljust(25), expected)
