#importing libraries
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pandas as pd

#reading training data
df=pd.read_csv('../../Dataset/DS1-train.csv', sep=',',header=None)
X = df.values
X = np.array(X)

Y = X[:,20:]
X = X[:,:20]

#fitting linear regression model to both indicator variables
reg = linear_model.LinearRegression()
reg.fit (X, Y) 
#creating array of coefficients
W = np.append(np.array([reg.intercept_]), np.array(reg.coef_).T, axis=0)
#turning in coefficients in a csv file
my_df = pd.DataFrame(W)
my_df.to_csv('coeffs.csv', index=False, header=False)

#Reading test data
df=pd.read_csv('../../Dataset/DS1-test.csv', sep=',',header=None)
X_test = df.values
X_test = np.array(X_test)
#splitting test data into data points and indictor variables
Y_test = X_test[:,20:]
X_test = X_test[:,:20]
#predicting using Linear Regression object
Y_predicted = reg.predict(X_test)
#finding argmax to predict class
Y_predicted_arr = np.argmax(Y_predicted, axis =1)
Y_true = np.argmax(Y_test, axis =1)

#Calculating best fit parameters
accuracy = 100.0*accuracy_score(Y_true, Y_predicted_arr)
precision = 100.0*precision_score(Y_true, Y_predicted_arr, average='binary')
recall = 100.0*recall_score(Y_true, Y_predicted_arr, average='binary')
F = 100.0*f1_score(Y_true, Y_predicted_arr, average='binary')

print "Accuracy %.2f %%\n Precision %.2f %%\n Recall %.2f %%\n F-measure %.2f %%\n" %(accuracy, precision, recall, F)
print "Coefficients have been turned in, in a coeffs.csv file, with the intercept being the first coefficient."

