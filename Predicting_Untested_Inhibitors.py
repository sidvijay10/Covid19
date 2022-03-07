

# Importing the libraries
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from statistics import mean


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

kinase_list = pd.read_csv('recursive_elimination_kinases_covid.csv')
kinase_list = kinase_list.values.tolist()

kinases = []
for kinase in kinase_list:
    kinases.append(kinase[0])
    
# Importing the dataset
X = dataset2[kinases].values
y = dataset2.iloc[:, 298].values

#Initialize hyperparameters in model to optimized hyperparameter values
classifier = Sequential()
classifier.add(Dense(units = 100, kernel_initializer = 'TruncatedNormal', activation = 'relu', input_dim = len(kinases)))
classifier.add(Dense(units = 100, kernel_initializer = 'TruncatedNormal', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'TruncatedNormal' )) 
classifier.compile(loss = 'mean_squared_error', optimizer='adam')
classifier.fit(X,y, epochs=80, batch_size=44)

X_predict = alldrugs2
X_predict = X_predict[kinases]
prediction_index = X_predict.index.tolist()
X_predict = X_predict.iloc[:, 0:len(kinases)+2].values

# Predicting the Test set results
y_pred = classifier.predict(X_predict)
untested_inhibitor_prediction = pd.DataFrame(y_pred.tolist(), index = prediction_index)
ranked_inhibitors = untested_inhibitor_prediction.sort_values(by=[0])
ranked_inhibitors.to_excel("Ranked_inhibitors_Covid.xlsx", sheet_name='1', header = ["Prediction"])



