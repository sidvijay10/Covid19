

# Importing the libraries
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error

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
X2 = dataset2[kinases].values
y2 = dataset2.iloc[:, 298].values

#Initialize hyperparameters in cross_val() to optimized hyperparameter values
def cross_val():
    y_pred_all = []
    y_test_all = []
    for num in range(len(X2)):
        y_test = y2[num]
        X_test = X2[num, :]
        X_test = np.array([X_test])
        X_test.T
        X_train = np.delete(X2, (num), axis=0)
        y_train = np.delete(y2, (num), axis=0)
        classifier = Sequential()
        classifier.add(Dense(units = 100, kernel_initializer = 'TruncatedNormal', activation = 'relu', input_dim = len(kinases)))
        classifier.add(Dense(units = 100, kernel_initializer = 'TruncatedNormal', activation = 'relu'))
        classifier.add(Dense(units = 1, kernel_initializer = 'TruncatedNormal' )) 
        classifier.compile(loss = 'mean_squared_error', optimizer='adam')
        classifier.fit(X_train, y_train, batch_size = 44, epochs = 80)
        y_pred = classifier.predict(X_test)
        y_pred_all.append(y_pred[0][0])
        y_test_all.append(y_test)

    return (y_pred_all)

y_pred_df = pd.DataFrame(cross_val())

for num in range(19):
    single_y_pred = cross_val()
    y_pred_df[str(num)] = single_y_pred
    
y_pred_df = y_pred_df.mean(axis = 1)
y_pred = y_pred_df.values.tolist()

print(mean_squared_error(y2.tolist(), y_pred))

    
