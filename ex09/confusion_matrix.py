import numpy as np
from sklearn.metrics import confusion_matrix

def confusion_matrix_(y:np.ndarray, yhat:np.ndarray, labels=None, df_option=True):
    """
    ...Docstring...
    """
    if labels is None:
        labels = np.unique(y).astype(object)
    confusion_matrix = pd.DataFrame(data=np.zeros((labels.shape[0], labels.shape[0])),
                                    index=labels,
                                    columns=labels)
    for index in labels:
        mask = y == index
        for col in labels:
            nb = np.sum(yhat[mask] == col)
            confusion_matrix[col][index] = nb
    if df_option == True:
        return confusion_matrix
    else:
        return confusion_matrix.values


if __name__ == '__main__':
    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'bird'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet'])
    # Example 1:
    ## your implementation
    confusion_matrix_(y, y_hat)
    ## Output:
    expected = np.array([[0, 0, 0], [0, 2, 1], [1, 0, 2]])
    ## sklearn implementation
    confusion_matrix(y, y_hat)
    ## Output:
    expected = np.array([[0, 0, 0], [0, 2, 1], [1, 0, 2]])
    # Example 2:
    ## your implementation
    confusion_matrix_(y, y_hat, labels=['dog', 'norminet'])
    ## Output:
    expected = np.array([[2, 1],[0, 2]])
    ## sklearn implementation
    confusion_matrix(y, y_hat, labels=['dog', 'norminet'])
    ## Output:
    expected = np.array([[2, 1], [0, 2]])
    # Example 3:
    confusion_matrix_(y_true, y_hat, df_option=True)
    # Output:
    #         bird  dog norminet
    # bird     0     0    0
    # dog      0     2    1
    # norminet 1     0    2
    # Example 4:
    confusion_matrix_(y_true, y_hat, labels=['bird', 'dog'], df_option=True)
    # Output:
    #     bird dog
    # bird 0    0
    # dog  0    2