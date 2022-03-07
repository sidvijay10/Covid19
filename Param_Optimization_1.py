
# Importing the libraries
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

response_data2 = pd.read_csv('IL1B_response.csv') # USE FOR IL1B DATA
#response_data2 = pd.read_csv('IL6_response.csv') # USE FOR IL6 DATA
#response_data2 = pd.read_csv('IL8_response.csv') # USE FOR IL8 DATA
#response_data2 = pd.read_csv('IL10_response.csv') # USE FOR IL10 DATA
#response_data2 = pd.read_csv('CCL2_response.csv') # USE FOR CCL2 DATA
#response_data2 = pd.read_csv('TNFalpha_response.csv') # USE FOR TNFalpha DATA
#response_data2 = pd.read_csv('GMCSF_response.csv') # USE FOR GMCSF DATA

drug_list2 = response_data2.iloc[:, 0].values
alldrugs2 = pd.read_csv('kir_allDrugs_namesDoses.csv', encoding='latin1')

alldrugs2 = alldrugs2.set_index('compound')
dataset2 = alldrugs2.loc[drug_list2]
response2 = response_data2['Response'].values
dataset2["response"] = response2

# Importing the dataset
X2 = dataset2.iloc[:, 0:298].values
y2 = dataset2.iloc[:, 298].values


def build_classifier(optimizer, init, activation, hl):
    classifier = Sequential()
    classifier.add(Dense(units = hl, kernel_initializer = init, activation = activation, input_dim = 298))
    classifier.add(Dense(units = hl, kernel_initializer = init, activation = activation))
    classifier.add(Dense(units = 1, kernel_initializer = init))
    classifier.compile(loss = 'mean_squared_error', optimizer= optimizer, metrics=[ 'mse'])
    return classifier
model = KerasRegressor(build_fn=build_classifier)
param_grid = {'batch_size': [1, 2, 4, 8, 16, 32, 44], 'init': ['normal'],
              'epochs': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400], 
              'activation': ['relu'], 
              'optimizer': ['adam'], 
              'hl': [100]}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs= 8, cv=44)
grid_result = grid.fit(X2, y2)

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']


for mean, stdev, param in zip(means, stds, params):
    print("Mean: %f (StDev: %f) with: %r" % (mean, stdev, param))
        
best_parameters = grid_result.best_params_
best_accuracy = grid_result.best_score_
print("\nBest Parameters: ")
for k, v in best_parameters.items():
    print(k, v)

