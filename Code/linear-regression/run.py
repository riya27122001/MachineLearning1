#importing header files
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model

#reading CandC_complete dataset
df=pd.read_csv('../q4/CandC_complete.csv', sep=',',header=None)
X = df.values
X = np.array(X)
#i iterates over 5 datasets
for i in xrange(1,6):
	#splitting into testing and training data
	X_train, X_test = train_test_split(X, test_size=0.2)
	#exporting test-train data to csv files
	my_df = pd.DataFrame(X_test)
	my_df.to_csv('CandC-test'+str(i)+'.csv', index=False, header=False)
	my_df = pd.DataFrame(X_train)
	my_df.to_csv('CandC-train'+str(i)+'.csv', index=False, header=False)
	X_train = np.array(X_train)
	X_test = np.array(X_test)
	#separating regression O/P parameter as y
	Y_train = X_train[:,-1:]
	X_train = X_train[:,:-1]
	Y_test = X_test[:,-1:]
	X_test = X_test[:,:-1]
	#linear regression fitting data
	regr = linear_model.LinearRegression()
	regr.fit(X_train, Y_train)
	#exporting coefficients to csv file
	W = np.append(np.array([regr.intercept_]), np.array(regr.coef_).T, axis=0)
	my_df = pd.DataFrame(W)
	my_df.to_csv('coeff_q5_'+str(i)+'.csv', index=False, header=False)
	#predicting on test data
	Y_pred = regr.predict(X_test)
	#computing and reporting RSS
	print("RSS%d: %.2f \n" %(i, ((Y_pred - Y_test)**2).sum()) )


#Mean square error turned out to be same, write why in report