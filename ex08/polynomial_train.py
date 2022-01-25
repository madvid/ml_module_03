import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.use('Qt5Agg')
#mpl.use('TkAgg')

path = os.path.join(os.path.dirname(__file__), '..', 'ex05')
sys.path.insert(1, path)
from mylinearregression import MyLinearRegression as MyLR

path = os.path.join(os.path.dirname(__file__), '..', 'ex07')
sys.path.insert(1, path)
from polynomial_model import add_polynomial_features


if __name__ == "__main__":
    # Retrieving and checking the data:
    try:
        data = pd.read_csv('are_blue_pills_magics.csv')
        
        # Checking column names
        expected_cols = ['Patient', 'Micrograms', 'Score']
        if not all([c in expected_cols for c in data.columns]):
            print("Missing or unexpected columns.", file = sys.stderr)
            sys.exit()
        # Checking of the dtype, strict test but it may helps to avoid
        # traps of twisted evaluators
        if not all([dt.kind in ['i', 'f'] for dt in data.dtypes]):
            print("Incorrect datatype for one or more columns.", file = sys.stderr)
            sys.exit()
    except:
        sys.exit()

    target = data.Score.values.reshape(-1,1)
    x = data.Micrograms.values
    
    x1 = add_polynomial_features(x, 1)
    x2 = add_polynomial_features(x, 2)
    x3 = add_polynomial_features(x, 3)
    x4 = add_polynomial_features(x, 4)
    x5 = add_polynomial_features(x, 5)
    x6 = add_polynomial_features(x, 6)
    
    theta4 = np.array([-20., 160., -80., 10., -1.]).reshape(-1,1)
    theta5 = np.array([1140., -1850., 1110., -305., 40., -2.]).reshape(-1,1)
    theta6 = np.array([9110., -18015., 13400., -4935., 966., -96.4, 3.86]).reshape(-1,1)
    
    np.random.seed = 42
    mylr1 = MyLR(np.random.rand(2, 1), alpha=1e-3, max_iter=1000000)
    mylr2 = MyLR(np.random.rand(3, 1), alpha=1e-3, max_iter=1000000)
    mylr3 = MyLR(np.random.rand(4, 1), alpha=6e-5, max_iter=5000000)
    mylr4 = MyLR(theta4, alpha=1e-6, max_iter=1000000)
    mylr5 = MyLR(theta5, alpha=4e-8, max_iter=1000000)
    mylr6 = MyLR(theta6, alpha=1e-9, max_iter=5000000)
    
    print("starting training model #1")
    mylr1.fit_(x1, target)
    print("starting training model #2")
    mylr2.fit_(x2, target)
    print("starting training model #3")
    mylr3.fit_(x3, target)
    print("starting training model #4")
    mylr4.fit_(x4, target)
    print("starting training model #5")
    mylr5.fit_(x5, target)
    print("starting training model #6")
    mylr6.fit_(x6, target)
    

    _, axe = plt.subplots(1, 1, figsize=(15, 8))
    
    x_sampling = np.linspace(1, 7, 100)
    axe.scatter(x, target, label='raw', c='black')
    axe.plot(x_sampling, mylr1.predict_(add_polynomial_features(x_sampling, 1)), label='pred mylr1')
    axe.plot(x_sampling, mylr2.predict_(add_polynomial_features(x_sampling, 2)), label='pred mylr2')
    axe.plot(x_sampling, mylr3.predict_(add_polynomial_features(x_sampling, 3)), label='pred mylr3')
    axe.plot(x_sampling, mylr4.predict_(add_polynomial_features(x_sampling, 4)), label='pred mylr4')
    axe.plot(x_sampling, mylr5.predict_(add_polynomial_features(x_sampling, 5)), label='pred mylr5')
    axe.plot(x_sampling, mylr6.predict_(add_polynomial_features(x_sampling, 6)), label='pred mylr6')
    plt.grid()
    plt.legend()
    plt.show()
    
    # MSE repport:
    print("#" * 50, "\nLoss report:")
    for idx, xx, model in zip(range(1,7), [x1, x2, x3, x4, x5, x6], [mylr1, mylr2, mylr3, mylr4, mylr5, mylr6]):
        print(f"mylr{idx}: MSE = {model.loss_(xx, target):.5f}")
    print("#" * 50)