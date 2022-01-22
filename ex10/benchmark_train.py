import os
import sys
import numpy as np
import pandas as pd

path = os.path.join(os.path.dirname(__file__), '..', 'ex09')
sys.path.insert(1, path)
from data_spliter import data_spliter

path = os.path.join(os.path.dirname(__file__), '..', 'ex07')
sys.path.insert(1, path)
from polynomial_model import add_polynomial_features


path = os.path.join(os.path.dirname(__file__), '..', 'ex05')
sys.path.insert(1, path)
from mylinearregression import MyLinearRegression

lst_feat = ["weight", "prod_distance", "time_delivery"]
lst_dataset = lst_feat + ["target"]

if __name__ == "__main__":
    # Importation of the dataset + basic checking:
    try:
        data = pd.read_csv("space_avocado.csv", index_col=0, dtype=np.float64)
    except:
        print("Issue when trying to retrieve the dataset.", file=sys.stderr)
        sys.exit()

    
    if any([not c in lst_dataset for c in data.columns]):
        print("At least a missing expected columns.", file=sys.stderr)
        sys.exit()

    if any([not dt.kind in ['i', 'f'] for dt in data.dtypes]):
        print("At least one column is not of expected kind dtype.", file=sys.stderr)
        sys.exit()
    
    x_train, x_test, y_train, y_test = data_spliter(data[lst_feat].values,
                                                     data["target"].values.reshape(-1,1), 0.8)

    print("x_train.shape = ", x_train.shape)
    print("x_test.shape = ", x_test.shape)
    print("y_train.shape = ", y_train.shape)
    print("y_test.shape = ", y_test.shape)