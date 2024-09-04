import sys

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

df: DataFrame = pd.read_csv('./data_pso_Cu.dat', header=None, sep='\s+')
df.columns = ['c1', 'c2', 'c3', 'c4', 'c5', 'Bulk', 'C11', 'C12', 'C44', 'Young', 'USF']
# sum from V1 to V5
df['sum'] = df['c1'] + df['c2'] + df['c3'] + df['c4'] + df['c5']
df = df.copy(deep=True)
df['v1'] = df['c1'] + 5
df['v2'] = df['c2'] + 5
df['v3'] = df['c3'] + 5
df['v4'] = df['c4'] + 5
df['v5'] = df['c5'] + 5

df = df.sort_values(by=['USF'], ascending=False)
# remove the data where USF is 1000
df = df[df['USF'] < 1000]

n = 0.98  # %
A = 43.5152e-10 * 25.1235e-10  # Copper
eV_to_J = 1.6021766208e-19
df.loc[:, 'USF_coverted'] = df['USF'] / A * eV_to_J * 1000

df_usf_use = df.copy(deep=True)

df_usf = df_usf_use.head(10000).copy()

feature_cols = ['USF_coverted']
label_cols = ['v1', 'v2', 'v3', 'v4', 'v5']
y = df_usf.loc[:, feature_cols]
X = df_usf.loc[:, label_cols]
print('y_max', y.USF_coverted.max())

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=34)  # train_size = 0.75
scaler = preprocessing.StandardScaler().fit(X_train)
# save the scaler
# joblib.dump(scaler, 'Cu_USF_scaler.pkl')
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# import MLPRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import SGDRegressor, BayesianRidge


# Create MLPRegressor
model = MLPRegressor(max_iter=10000)

models = [
    # ('RF', RandomForestRegressor()),
    ('SGD', SGDRegressor()),
    ('MLP', MLPRegressor(max_iter=10000)),
    ('BR', BayesianRidge()),
]

stacking_regressor = StackingRegressor(estimators=models, final_estimator=MLPRegressor(max_iter=10000))

model = stacking_regressor


# Train the model using the training sets
model.fit(X_train, y_train)


# R2 score
y_pred = model.predict(X_test)
print('R2 score', r2_score(y_test, y_pred.ravel()))


# In[ ]:

# Shap
# use shap to explain the prediction of the first element
import shap

# select a set of background examples to take an expectation over
background = X_train
predict_function = lambda x: model.predict(x)
# explain predictions of the model on four images
e = shap.Explainer(predict_function, background)

# Select a specific instance for explanation
instance_to_explain = X_test

# Compute SHAP values for the selected instance
shap_values = e(instance_to_explain)

# joblib save shap_values
joblib.dump(shap_values, 'Cu_USF_shap_values_default.pkl')

# In[]:


feature_names = ['Cu', 'Cr', 'Co', 'Fe', 'Ni']

shap.plots.violin(shap_values, feature_names=feature_names)
