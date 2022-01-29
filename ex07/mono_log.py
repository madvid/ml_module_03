import sys
import os
import pandas as pd
import numpy as np

path = os.path.join(os.path.dirname(__file__), '..', 'ex07')
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
# ########################################################################## #
#  ______________________________ FUNCTIONS ________________________________ #
# ########################################################################## #

def usage():
    s = "usgage: python mono_log.py -zipcode=0/1/2/3"
    print(s)

def labelbinarizer(y, target):
    y_ = np.zeros(y.shape, dtype='int8')
    y_[np.where(y == target)] = 1
    return y_

# ########################################################################## #
#  ________________________________ MAIN ___________________________________ #
# ########################################################################## #

if __name__ == "__main__":
    # Parsing the arguments:
    parameters = sys.argv()
    if len(parameters == 1):
        usage()
        sys.exit()

    if len(parameters != 2):
        s = "Only `-zipcode=x` is exptected."
        print(s, sys.stderr)
        sys.exit()

    split = parameters.split('=')
    if len(split != 2) or (split[0] != '-zipcode') or (split[1] not in s_nb):
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
        # casting the y data:
        y.astype('int8')
    except:
        s = "Something wrong when casting data to integer."
        print(s, file=sys.stderr)
        sys.exit()

    if x.shape[0] != y.shape[0]:
        s = f"Unmatching number of lines between {file_x} and {file_y}"
        print(s, file=sys.stderr)
        sys.exit()
    
    y_ = labelbinarizer(y)
    
    x_train, x_test, y_train, y_test = data_spliter(x.values, y_, 0.2)
    
    scaler_x = MyStandardScaler()
    scaler_y = MyStandardScaler()
    scaler_x.fit(x_train)
    scaler_y.fit(y_train)
    x_train_tr = scaler_x.transform(x_train)
    y_train_tr = scaler_y.transform(y_train)
    x_test_tr = scaler_x.transform(x_test)
    y_test_tr = scaler_y.transform(y_test)

    monolr = MyLogR(np.random.rand(x.shape[1], 1), alpha=1e-2, max_iter=10000)
    monolr.fit(x_train_tr, y_train_tr)


