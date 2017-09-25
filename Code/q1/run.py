
import numpy as np
import csv
import random
import pandas as pd
from sklearn.model_selection import train_test_split

#generatng random PSD semidefinite matrix to be used as covariance matrix for both classes
cov = np.random.randint(0.0,10.0,size=(20,20))
cov_final = np.dot(cov,cov.T)
#average standard deviation of all parameters
dev = ((np.sqrt(cov_final.diagonal())).sum())/20.0

#centroids for two classes with a distance gap of 0.4*dev
mean = np.zeros(20)
mean1 = np.zeros(20)
mean1[0] = -0.4*dev

#generating two normally distributed classes
#parameters __
#centroids - mean, mean1
#covariance matrix - cov_final
#no.of samples to be generated - 2000
X = np.random.multivariate_normal(mean, cov_final, 2000)
Y = np.random.multivariate_normal(mean1, cov_final, 2000)
X = np.array(X)
Y = np.array(Y)

#appending indicator variables
#column 1 indicates if element belongs to class -- 1, 1 if it does, 0 if it doesn't
#column 2 indicates if element belongs to class -- 2, 1 if it does, 0 if it doesn't
z = np.zeros((2000,1))
o = np.ones((2000,1))
X = np.append(X, o, axis=1)
X = np.append(X, z, axis=1)
Y = np.append(Y, z, axis=1)
Y = np.append(Y, o, axis=1)

# splitting into training and test data --  70-30 ratio
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3) #random_state=42 for retrieving data?
T_test = np.append(X_test, Y_test, axis=0)
T_train = np.append(X_train, Y_train, axis=0)
T_test = np.array(T_test)
T_train = np.array(T_train)

#exporting test and training data into csv file
my_df = pd.DataFrame(T_test)
my_df.to_csv('DS1-test.csv', index=False, header=False)
my_df = pd.DataFrame(T_train)
my_df.to_csv('DS1-train.csv', index=False, header=False)

