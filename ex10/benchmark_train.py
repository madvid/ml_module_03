# ######################################################### #
# ######################################################### #

# THE PROCEDURE PERFORMED HERE TO SEEK TO THE BEST MODEL
# IS A VERY DUMMY ONE !!!
# NO STRATEGY HERE, I MEAN IT IS JUST BRAINLESS STRATEGY:
#      * Bunch of models (no use of correlation or mutual information!)
#      * training and evaluating, hoping for the best
#
# <<< /!\ DO NOT PERFROM MODEL SELECTION THIS WAY             >>>
# <<<     it is just to get XP school attached to the project >>>

# The only small interesting thing here is the multiprocessing ...
# ######################################################### #
# ######################################################### #

# ######################################################### #
#                    LIBRARIES IMPORT                       #
# ######################################################### #

import os
import sys
import numpy as np
import pandas as pd

# Pour le multiprocessing
import concurrent.futures
from multiprocessing import cpu_count

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

n_cpu = cpu_count()
cpu_use = int(n_cpu/2)
print("CPU USE: ", cpu_use)

lst_check_feat = ["weight", "prod_distance", "time_delivery"]
lst_dataset = lst_check_feat + ["target"]

col2idx = {'w' : 0,
           'p' : 1,
           't' : 2,
           'w2' : 3,
           'p2' : 4,
           't2' : 5,
           'w3' : 6,
           'p3' : 7,
           't3' : 8,
           'w4' : 9,
           'p4' : 10,
           't4' : 11,
           'wp' : 12,
           'w2p' : 13,
           'wp2' : 14,
           'w2p2' : 15,
           'wt' : 16,
           'w2t' : 17,
           'wt2' : 18,
           'w2t2' : 19,
           'pt' : 20,
           'p2t' : 21,
           'pt2' : 22,
           'p2t2' : 23}


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
    
    # ###################################################################### #
    #                            First Bath of models                        #
    # ###################################################################### #
    # Simple models:
    lr_w = MyLR(np.random.rand(2,1), alpha=1e-2, max_iter=50000)
    lr_p = MyLR(np.random.rand(2,1), alpha=1e-2, max_iter=50000)
    lr_t = MyLR(np.random.rand(2,1), alpha=1e-2, max_iter=50000)
    
    lr_wp = MyLR(np.random.rand(3,1), alpha=1e-2, max_iter=50000)
    lr_wt = MyLR(np.random.rand(3,1), alpha=1e-2, max_iter=50000)
    lr_pt = MyLR(np.random.rand(3,1), alpha=1e-2, max_iter=50000)
    
    lr_wpt = MyLR(np.random.rand(4,1), alpha=1e-2, max_iter=50000)
    

    simple_models = [lr_w, lr_p, lr_t, lr_wp, lr_wt, lr_pt, lr_wpt]
    lst_vars = [['w'], ['p'], ['t'],
                ['w', 'p'], ['w', 't'], ['p', 't'], ['w', 'p', 't']]
    
    
    # ii = 1
    # for model, vars in zip(simple_models, lst_vars):
    #     print(f"Batch simple models (model {ii} / 7)")
    #     model.fit_(x_train_tr[:,data_idx(vars)], y_train_tr)
    #     ii += 1
    
    batch1_trained = []
    nb = len(simple_models)
    s_state = ['[ ]'] * nb
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_use) as executor:
        # Starting/Exectuting processes
        for ii, model, vars in zip(range(nb), simple_models, lst_vars):
            print(f"Batch simple: starting model {ii + 1} / {nb}", end='\r', flush=True)
            model._tag_, model._idx_ = f"batch_1_model_{ii + 1}", ii
            batch1_trained.append(executor.submit(model.fit_,
                                           x_train_tr[:,data_idx(vars)],
                                           y_train_tr))
        print('\n')
        # Action when process are completed
        # (just printing the state string to have an idea of the remaining train)
        for task in concurrent.futures.as_completed(batch1_trained):
            if task.result() != None:
                s_state[task.result()._idx_] = '[✔]'
                print('Simple batch: ' + ' '.join(s_state), end='\r', flush=True)
    
    # ###################################################################### #
    #                           Second Bath of models                        #
    # ###################################################################### #
    # 'intermediate' models
    lr_w2 = MyLR(np.random.rand(3, 1), alpha=1e-2, max_iter=50000)
    lr_w3 = MyLR(np.random.rand(4, 1), alpha=1e-2, max_iter=50000)
    lr_w4 = MyLR(np.random.rand(5, 1), alpha=1e-2, max_iter=50000)
    
    lr_p2 = MyLR(np.random.rand(3, 1), alpha=1e-2, max_iter=50000)
    lr_p3 = MyLR(np.random.rand(4, 1), alpha=1e-2, max_iter=50000)
    lr_p4 = MyLR(np.random.rand(5, 1), alpha=1e-2, max_iter=50000)
    
    lr_t2 = MyLR(np.random.rand(3, 1), alpha=1e-2, max_iter=50000)
    lr_t3 = MyLR(np.random.rand(4, 1), alpha=1e-2, max_iter=50000)
    lr_t4 = MyLR(np.random.rand(5, 1), alpha=1e-2, max_iter=50000)
    
    lr_w_p_2 = MyLR(np.random.rand(5, 1), alpha=1e-2, max_iter=50000)
    lr_w_p_3 = MyLR(np.random.rand(7, 1), alpha=1e-2, max_iter=50000)
    lr_w_p_4 = MyLR(np.random.rand(9, 1), alpha=1e-2, max_iter=50000)
    
    lr_w_t_2 = MyLR(np.random.rand(5, 1), alpha=1e-2, max_iter=50000)
    lr_w_t_3 = MyLR(np.random.rand(7, 1), alpha=1e-2, max_iter=50000)
    lr_w_t_4 = MyLR(np.random.rand(9, 1), alpha=1e-2, max_iter=50000)
    
    lr_p_t_2 = MyLR(np.random.rand(5, 1), alpha=1e-2, max_iter=50000)
    lr_p_t_3 = MyLR(np.random.rand(7, 1), alpha=1e-2, max_iter=50000)
    lr_p_t_4 = MyLR(np.random.rand(9, 1), alpha=1e-2, max_iter=50000)
    
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
    
    
    # I could do a wrapping function taking only the x_train_tr, y_train_tr and models
    # and performing the full multi process training
    batch2_trained = []
    nb = len(intermediate_models)
    s_state = ['[ ]'] * nb
    print('\n')
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_use) as executor:
        
        for ii, model, vars in zip(range(nb), intermediate_models, lst_vars):
            print(f"Batch intermediate: starting model {ii + 1} / {nb}", end='\r', flush=True)
            model._tag_, model._idx_ = f"batch_2_model_{ii + 1}", ii
            batch2_trained.append(executor.submit(model.fit_, x_train_tr[:,data_idx(vars)], y_train_tr))
        print('\n')
        for task in concurrent.futures.as_completed(batch2_trained):
            if task.result() != None:
                s_state[task.result()._idx_] = '[✔]'
                print('Intermediate batch: ' + ' '.join(s_state), end='\r', flush=True)
    
    # ###################################################################### #
    #                            Third Bath of models                        #
    # ###################################################################### #
    # 'sophisticate' models
    # models with ∑w^n n in (1,...,4) and ∑p^n n in (1,...,4) plus
    # the terms ∑t^m. 1st: t^1 - 2nd: t^1 + p^2 - 3rd: p^1 + t^2 + t^3
    lr_WP_t1 = MyLR(np.random.rand(10, 1), alpha=1e-2, max_iter=50000)
    lr_WP_t2 = MyLR(np.random.rand(11, 1), alpha=1e-2, max_iter=50000)
    lr_WP_t3 = MyLR(np.random.rand(12, 1), alpha=1e-2, max_iter=50000)
    lr_WP_t4 = MyLR(np.random.rand(13, 1), alpha=1e-2, max_iter=50000)
    
    # models with ∑w^n n in (1,...,4) and ∑t^n n in (1,...,4) plus
    # the terms ∑p^m. 1st: p^1 - 2nd: p^1 + p^2 - 3rd: p^1 + p^2 + p^3
    lr_WT_p1 = MyLR(np.random.rand(10, 1), alpha=1e-2, max_iter=50000)
    lr_WT_p2 = MyLR(np.random.rand(11, 1), alpha=1e-2, max_iter=50000)
    lr_WT_p3 = MyLR(np.random.rand(12, 1), alpha=1e-2, max_iter=50000)
    
    # models with ∑p^n n in (1,...,4) and ∑t^n n in (1,...,4) plus
    # the terms ∑w^m. 1st: w^1 - 2nd: w^1 + w^2 - 3rd: w^1 + w^2 + w^3
    lr_PT_w1 = MyLR(np.random.rand(10, 1), alpha=1e-2, max_iter=50000)
    lr_PT_w2 = MyLR(np.random.rand(11, 1), alpha=1e-2, max_iter=50000)
    lr_PT_w3 = MyLR(np.random.rand(12, 1), alpha=1e-2, max_iter=50000)
    
    # models with ∑(w^n+ p^n + t^n) n in (1,...,4) plus
    # one cross term: 'wp', w^2p, wp^2, w^2p^2
    # and finally all the cross terms 'wp' + w^2p + wp^2 + w^2p^2
    lr_WPT_wp = MyLR(np.random.rand(14, 1), alpha=1e-2, max_iter=50000)
    lr_WPT_w2p = MyLR(np.random.rand(14, 1), alpha=1e-2, max_iter=50000)
    lr_WPT_wp2 = MyLR(np.random.rand(14, 1), alpha=1e-2, max_iter=50000)
    lr_WPT_w2p2 = MyLR(np.random.rand(14, 1), alpha=1e-2, max_iter=50000)
    lr_WPT_WP = MyLR(np.random.rand(17, 1), alpha=1e-2, max_iter=50000)
    
    # models with ∑(w^n+ p^n + t^n) n in (1,...,4) plus
    # one cross term: 'wt', w^2t, wt^2, w^2t^2
    # and finally all the cross terms 'wt' + w^2t + wt^2 + w^2t^2
    lr_WPT_wt = MyLR(np.random.rand(14, 1), alpha=1e-2, max_iter=50000)
    lr_WPT_w2t = MyLR(np.random.rand(14, 1), alpha=1e-2, max_iter=50000)
    lr_WPT_wt2 = MyLR(np.random.rand(14, 1), alpha=1e-2, max_iter=50000)
    lr_WPT_w2t2 = MyLR(np.random.rand(14, 1), alpha=1e-2, max_iter=50000)
    lr_WPT_WT = MyLR(np.random.rand(17, 1), alpha=1e-2, max_iter=50000)
    
    # models with ∑(w^n+ p^n + t^n) n in (1,...,4) plus
    # one cross term: 'pt', p^2t, pt^2, p^2t^2
    # and finally all the cross terms 'pt' + p^2t + pt^2 + p^2t^2
    lr_WPT_pt = MyLR(np.random.rand(14, 1), alpha=1e-2, max_iter=50000)
    lr_WPT_p2t = MyLR(np.random.rand(14, 1), alpha=1e-2, max_iter=50000)
    lr_WPT_pt2 = MyLR(np.random.rand(14, 1), alpha=1e-2, max_iter=50000)
    lr_WPT_p2t2 = MyLR(np.random.rand(14, 1), alpha=1e-2, max_iter=50000)
    lr_WPT_PT = MyLR(np.random.rand(17, 1), alpha=1e-2, max_iter=50000)
    
    # models with ∑(w^n+ p^n + t^n) n in (1,...,4) plus
    # all the cross term of wp, wt and pt
    lr_WPT_WP_WT_PT = MyLR(np.random.rand(25, 1), alpha=1e-2, max_iter=50000)
    
    sophisticatish_models = [lr_WP_t1, lr_WP_t2, lr_WP_t3, lr_WP_t4,
                           lr_WT_p1, lr_WT_p2, lr_WT_p3,
                           lr_PT_w1, lr_PT_w2, lr_PT_w3,
                           lr_WPT_wp, lr_WPT_w2p, lr_WPT_wp2, lr_WPT_w2p2,
                           lr_WPT_WP, lr_WPT_wt, lr_WPT_w2t, lr_WPT_wt2,
                           lr_WPT_w2t2, lr_WPT_WT, lr_WPT_pt, lr_WPT_p2t,
                           lr_WPT_pt2, lr_WPT_p2t2, lr_WPT_PT]

    lst_W = ['w', 'w2', 'w3', 'w4']
    lst_T = ['t', 't2', 't3', 't4']
    lst_P = ['p', 'p2', 'p3', 'p4']
    
    lst_WP = lst_W + lst_P
    lst_WT = lst_W + lst_T
    lst_PT = lst_P + lst_T
    lst_WPT = lst_W + lst_P + lst_T

    # My apologize, it might be painful to read: It is the list of var string    
    lst_vars = [lst_WP + [lst_T[0]], lst_WP + lst_T[0:2], lst_WP + lst_T[:3],
                lst_WPT, lst_WT + [lst_P[0]], lst_WT + lst_P[0:2], lst_WT + lst_P[:3],
                lst_PT + [lst_W[0]], lst_PT + lst_W[0:2], lst_PT + lst_W[:3],
                lst_WPT + ['wp'], lst_WPT + ['w2p'], lst_WPT + ['wp2'], lst_WPT + ['w2p2'],
                lst_WPT + ['wp', 'w2p', 'wp2', 'w2p2'],
                lst_WPT + ['wt'], lst_WPT + ['w2t'], lst_WPT + ['wt2'], lst_WPT + ['w2t2'],
                lst_WPT + ['wt', 'w2t', 'wt2', 'w2t2'],
                lst_WPT + ['pt'], lst_WPT + ['p2t'], lst_WPT + ['pt2'], lst_WPT + ['p2t2'],
                lst_WPT + ['pt', 'p2t', 'pt2', 'p2t2'],
                lst_WPT + ['wp', 'w2p', 'wp2', 'w2p2'] + ['wt', 'w2t', 'wt2', 'w2t2'] \
                    + ['pt', 'p2t', 'pt2', 'p2t2']]
    for elem in lst_vars:
        print(elem)
    # Yep I could have done this function ...
    batch3_trained = []
    nb = len(sophisticatish_models)
    s_state = ['[ ]'] * nb
    print('\n')
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_use) as executor:
        
        for ii, model, vars in zip(range(nb), sophisticatish_models, lst_vars):
            print(f"Batch 'sophisticated': starting model {ii + 1} / {nb}", end='\r', flush=True)
            model._tag_, model._idx_ = f"batch_3_model_{ii + 1}", ii
            batch3_trained.append(executor.submit(model.fit_, x_train_tr[:,data_idx(vars)], y_train_tr))
        print('\n')
        for task in concurrent.futures.as_completed(batch3_trained):
            if task.result() != None:
                s_state[task.result()._idx_] = '[✔]'
                print("Sophisticatish batch: " + ' '.join(s_state), end='\r', flush=True)
    
    print('\n')
    
    