# ######################################################### #
#                    LIBRARIES IMPORT                       #
# ######################################################### #

import os
import sys
import numpy as np
import pandas as pd

path = os.path.join(os.path.dirname(__file__), '..', 'ex09')
sys.path.insert(1, path)
from data_spliter import data_spliter

path = os.path.join(os.path.dirname(__file__), '..', 'ex05')
sys.path.insert(1, path)
from mylinearregression import MyLinearRegression as MyLR

path = os.path.join(os.path.dirname(__file__), '..', 'ex07')
sys.path.insert(1, path)
from polynomial_model import add_polynomial_features

path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
from scaler import MyStandardScaler


# ######################################################### #
#                        CONSTANTES                         #
# ######################################################### #

lst_check_feat = ["weight", "prod_distance", "time_delivery"]
lst_dataset = lst_check_feat + ["target"]

col2idx = {'w': 0,
           'p': 1,
           't': 2,
           'w2': 3,
           'p2': 5,
           't2': 6,
           'w3': 7,
           'p3': 8,
           't3': 9,
           'w4': 10,
           'p4': 11,
           't4': 12,
           'wp': 13,
           'w2p': 14,
           'wp2': 15,
           'w2p2': 16,
           'wt': 17,
           'w2t': 18,
           'wt2': 19,
           'w2t2': 20,
           'p2t': 21,
           'pt2': 22,
           'p2t2': 23}

# ######################################################### #
#                  FUNCTION DEFINITIONS                     #
# ######################################################### #

def data_idx(cols):
    lst = []
    for c in cols:
        lst.append(col2idx[c])
    return np.array(lst, dtype=int)

# ######################################################### #
#                             MAIN                          #
# ######################################################### #

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
    
    data.rename(columns={'weight':'w', 'prod_distance':'p', 'time_delivery':'t'}, inplace=True)
    # Data augmentation: the subject specifies we need to use add_polynomial_features method:
    # But the use of numpy.polynomila.polynomial.polyvander2d would be wiser.
    w_1to4 = add_polynomial_features(data.w.values.reshape(-1,1), 4)
    p_1to4 = add_polynomial_features(data.p.values.reshape(-1,1), 4)
    t_1to4 = add_polynomial_features(data.t.values.reshape(-1,1), 4)
    
    for ii in range(1,4):
        data[f'w{ii + 1}'] = w_1to4[:,ii]
        data[f'p{ii + 1}'] = p_1to4[:,ii]
        data[f't{ii + 1}'] = t_1to4[:,ii]
        
    data['wp'] = data['w'] * data['p'] 
    data['w2p'] = (data['w'] ** 2) * data['p']
    data['wp2'] = data['w'] * (data['p'] ** 2)     
    data['w2p2'] = (data['w'] ** 2) * (data['p'] ** 2) 
    
    data['wt'] = data['w'] * data['t']
    data['w2t'] = (data['w'] ** 2) * data['t']
    data['wt2'] = data['w'] * (data['t'] ** 2) 
    data['w2t2'] = (data['w'] ** 2) * (data['t'] ** 2) 
    
    data['pt'] = data['p'] * data['t']
    data['p2t'] = (data['p'] ** 2) * data['t']
    data['pt2'] = data['p'] * (data['t'] ** 2) 
    data['p2t2'] = (data['p'] ** 2) * (data['t'] ** 2) 
    
    
    cols = data.columns.values
    cols = cols[cols != 'target']
    print(cols)
    x_train, x_test, y_train, y_test = data_spliter(data[cols].values,
                                                     data["target"].values.reshape(-1,1), 0.8)
    
    # We split the test set in 2: a real test set and cross validation set.
    # We should do it on the training set, especially if we would do k-fold CV,
    # but it will not be the case here
    sep = int(np.floor(0.5 * x_test.shape[0]))
    x_cross, y_cross = x_test.copy()[:sep], y_test.copy()[:sep]
    x_test, y_test = x_test.copy()[sep:], y_test.copy()[sep:]
    
    scaler_x = MyStandardScaler()
    scaler_y = MyStandardScaler()
    scaler_x.fit(x_train)
    scaler_y.fit(y_train)
    
    x_train_tr, y_train_tr = scaler_x.transform(x_train), scaler_y.transform(y_train)
    x_cross_tr, y_cross_tr = scaler_x.transform(x_cross), scaler_y.transform(y_cross)
    x_test_tr, y_test_tr = scaler_x.transform(x_test), scaler_y.transform(y_test)
    
    print("x_train.shape = ", x_train.shape)
    print("x_test.shape = ", x_test.shape)
    print("y_train.shape = ", y_train.shape)
    print("y_test.shape = ", y_test.shape)
    
    # ###################################################################### #
    #                            First Bath of models                        #
    # ###################################################################### #
    # Simple models:
    lr_w = MyLR(np.random.rand(2,1), alpha=1e-2, max_iter=5000)
    lr_p = MyLR(np.random.rand(2,1), alpha=1e-2, max_iter=5000)
    lr_t = MyLR(np.random.rand(2,1), alpha=1e-2, max_iter=5000)
    
    lr_wp = MyLR(np.random.rand(3,1), alpha=1e-2, max_iter=5000)
    lr_wt = MyLR(np.random.rand(3,1), alpha=1e-2, max_iter=5000)
    lr_pt = MyLR(np.random.rand(3,1), alpha=1e-2, max_iter=5000)
    
    lr_wpt = MyLR(np.random.rand(4,1), alpha=1e-2, max_iter=5000)
    
    simple_models = [lr_w, lr_p, lr_t, lr_wp, lr_wt, lr_pt, lr_wpt]
    lst_vars = [['w'], ['p'], ['t'], ['w', 'p'], ['w', 't'], ['p', 't'], ['w', 'p', 't']]
    
    ii = 1
    for model, vars in zip(simple_models, lst_vars):
        print(f"Batch simple models (model {ii} / 7)")
        model.fit_(x_train_tr[data_idx(vars)], y_train_tr)
        ii += 1
    
    # ###################################################################### #
    #                           Second Bath of models                        #
    # ###################################################################### #
    # 'intermediate' models
    lr_w2 = MyLR(np.random.rand(3, 1), alpha=1e-2, max_iter=5000)
    lr_w3 = MyLR(np.random.rand(4, 1), alpha=1e-2, max_iter=5000)
    lr_w4 = MyLR(np.random.rand(5, 1), alpha=1e-2, max_iter=5000)
    
    lr_p2 = MyLR(np.random.rand(3, 1), alpha=1e-2, max_iter=5000)
    lr_p3 = MyLR(np.random.rand(4, 1), alpha=1e-2, max_iter=5000)
    lr_p4 = MyLR(np.random.rand(5, 1), alpha=1e-2, max_iter=5000)
    
    lr_t2 = MyLR(np.random.rand(3, 1), alpha=1e-2, max_iter=5000)
    lr_t3 = MyLR(np.random.rand(4, 1), alpha=1e-2, max_iter=5000)
    lr_t4 = MyLR(np.random.rand(5, 1), alpha=1e-2, max_iter=5000)
    
    lr_w_p_2 = MyLR(np.random.rand(5, 1), alpha=1e-2, max_iter=5000)
    lr_w_p_3 = MyLR(np.random.rand(7, 1), alpha=1e-2, max_iter=5000)
    lr_w_p_4 = MyLR(np.random.rand(9, 1), alpha=1e-2, max_iter=5000)
    
    lr_w_t_2 = MyLR(np.random.rand(5, 1), alpha=1e-2, max_iter=5000)
    lr_w_t_3 = MyLR(np.random.rand(7, 1), alpha=1e-2, max_iter=5000)
    lr_w_t_4 = MyLR(np.random.rand(9, 1), alpha=1e-2, max_iter=5000)
    
    lr_p_t_2 = MyLR(np.random.rand(5, 1), alpha=1e-2, max_iter=5000)
    lr_p_t_3 = MyLR(np.random.rand(7, 1), alpha=1e-2, max_iter=5000)
    lr_p_t_4 = MyLR(np.random.rand(9, 1), alpha=1e-2, max_iter=5000)
    
    intermediate_models = [lr_w2, lr_w3, lr_w4, lr_p2, lr_p3, lr_p4, lr_t2,
                           lr_t3, lr_t4, lr_w_p_2, lr_w_p_3, lr_w_p_4, lr_w_t_2,
                           lr_w_t_3, lr_w_t_4, lr_p_t_2, lr_p_t_3, lr_p_t_4]
    
    lst_vars = [['w', 'w2'],
                ['w', 'w2', 'w3'],
                ['w', 'w2', 'w3', 'w4'],
                ['p', 'p2'],
                ['p', 'p2', 'p3'],
                ['p', 'p2', 'p3', 'p4'],
                ['t', 't2'],
                ['t', 't2', 't3'],
                ['t', 't2', 't3', 't4'],
                ['w', 'w2', 'p', 'p2'],
                ['w', 'w2', 'w3', 'p', 'p2', 'p3'],
                ['w', 'w2', 'w3', 'w4', 'p', 'p2', 'p3', 'p4'],
                ['w', 'w2', 't', 't2'],
                ['w', 'w2', 'w3', 't', 't2', 't3'],
                ['w', 'w2', 'w3', 'w4', 't', 't2', 't3', 't4'],
                ['p', 'p2', 't', 't2'],
                ['p', 'p2', 'p3', 't', 't2', 't3'],
                ['p', 'p2', 'p3', 'p4', 't', 't2', 't3', 't4']]
    
    ii = 1
    for model, vars in zip(intermediate_models, lst_vars):
        print(f"Batch simple models (model {ii} / 18)")
        model.fit_(x_train_tr[data_idx(vars)], y_train_tr)
        ii += 1
    
    # ###################################################################### #
    #                            Third Bath of models                        #
    # ###################################################################### #
    # 'sophisticate' models