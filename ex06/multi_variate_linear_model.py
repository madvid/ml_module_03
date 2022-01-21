import sys
import os
import numpy as np
import pandas as pd

path = os.path.join(os.path.dirname(__file__), '..', 'ex05')
sys.path.insert(1, path)
from mylinearregression import MyLinearRegression

path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
from plot import plot

if __name__ == '__main__':
    # ############################################################### #
    #             PART ONE: Univariate Linear Regression              #
    # ############################################################### #
    # Retrieving and checking the data:
    try:
        data = pd.read_csv('spacecraft_data.csv')
        
        # Checking column names
        expected_cols = ['Age', 'Thrust_power', 'Terameters', 'Sell_price']
        if not all([c in expected_cols for c in data.columns]):
            print("Missing or unexpected columns.", file = sys.stderr)
            sys.exit()
        # Checking of the dtype, strict test but it may helps to avoid
        # traps of twisted evaluators
        if not all([dt in [np.float64] for dt in data.dtypes]):
            print("Incorrect datatype for one or more columns.", file = sys.stderr)
            sys.exit()
    except:
        sys.exit()
        
    np.random.seed = 42
    mylr_Age = MyLinearRegression(np.random.rand(2,1), alpha = 5e-3, max_iter = 50000)
    mylr_Thrust = MyLinearRegression(np.random.rand(2,1), alpha = 1e-4, max_iter = 50000)
    mylr_Distance = MyLinearRegression(np.random.rand(2,1), alpha = 1e-4, max_iter = 750000)
    
    target = data.Sell_price.values.reshape(-1,1)
    age = data.Age.values.reshape(-1,1)
    thrust = data.Thrust_power.values.reshape(-1,1)
    distance = data.Terameters.values.reshape(-1,1)
    
    # Training: no splitting of the data into tain and test sets
    # (for those who knows as ML/DL students, datascientists, ... 
    # close your eyes, it will be over soon)
    mylr_Age.fit_(age, target)
    mylr_Thrust.fit_(thrust, target)
    mylr_Distance.fit_(distance, target)
    
    
    axes_labels = [r"$x_1$: age (in years)",
                   "y: sell price (in keuros)"]
    data_labels = {"raw":"Sell price", "prediction":"Predicted price"}
    colors = {"raw": "navy", "prediction": "royalblue"}
    title = f"Sell price vs Age -- MSE = {mylr_Age._loss_(target, mylr_Age.predict_(age))}"
    plot(age, target, mylr_Age.predict_(age), title,
         True, axes_labels, data_labels, colors)
    
    
    axes_labels = [r"$x_2$: thrust powers (in 10Km/s)",
                   "y: sell price (in keuros)"]
    colors = {"raw": "forestgreen", "prediction": "limegreen"}
    title = f"Sell price vs Thrust power -- MSE = {mylr_Thrust._loss_(target, mylr_Thrust.predict_(age))}"
    plot(thrust, target, mylr_Thrust.predict_(thrust), title,
         True, axes_labels, data_labels, colors)
    
    
    axes_labels = [r"$x_3$: distance totalizer value of spacecraft (in Tmeters)",
                   "y: sell price (in keuros)"]
    colors = {"raw": "indigo", "prediction": "violet"}
    title = f"Sell price vs Distance -- MSE = {mylr_Distance._loss_(target, mylr_Distance.predict_(age))}"
    plot(distance, target, mylr_Distance.predict_(distance), title,
         True, axes_labels, data_labels, colors)
    
    
    