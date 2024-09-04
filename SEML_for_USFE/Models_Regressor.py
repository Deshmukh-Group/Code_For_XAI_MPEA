import joblib
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB

from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import pandas as pd

import numpy as np
from numpy import absolute, mean, std


def regressor_compare(X_train: np.ndarray, y_train: pd.DataFrame, X_test: np.ndarray,
                      y_test: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight script to test many models
    :param X_train: training split (after Preprocessing)
    :param y_train: training target vector
    :param X_test: test split (after Preprocessing)
    :param y_test: test target vector
    :return: DataFrame of predictions
    """

    models = [
        #('RF', RandomForestRegressor()),
        ('SGD', SGDRegressor()),
        ('MLP', MLPRegressor(max_iter=10000)),
        ('BR', BayesianRidge()),
    ]

    stacking_regressor = StackingRegressor(estimators=models, final_estimator=MLPRegressor(max_iter=10000))

    results = []
    dfs = pd.DataFrame()
    for name, model in (models + [("SEM", stacking_regressor)]):
        # define the direct MultiOutput wrapper model
        wrapper = MultiOutputRegressor(model)
        # define the evaluation procedure
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate the model and collect the scores
        n_scores = cross_val_score(wrapper, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        # force the scores to be positive
        n_scores = absolute(n_scores)
        # summarize performance
        print('Model_Validation: %s ;' % name, 'MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
        results.append([name, mean(n_scores), std(n_scores)])

        df_temp = pd.DataFrame(results, columns=['Model_Name', 'MAE_mean', 'MAE_std'])
        dfs = pd.concat([dfs, df_temp], ignore_index=True)

        # Training and Testing
        wrapper.fit(X_train, y_train)
        y_pred = wrapper.predict(X_test)
        MAE = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
        R2 = r2_score(y_test, y_pred, multioutput='uniform_average')
        print('Model Name: %s ;' % name, 'Testing MAE: %.3f' % MAE, 'R2: %.3f' % R2)
        # save the model to disk
        joblib.dump(wrapper, 'model_' + name + '.pkl')

    return dfs
