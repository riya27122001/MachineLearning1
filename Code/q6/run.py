#importing header files
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model

b=[[]]
#j iterates over lamda from 0.5 to 50.0
for j in np.arange(0.5,50.0,0.5):
	a = []
	#i iterates over 5 datasets
	for i in xrange(1,6):
		#reading test and train datasets
		df=pd.read_csv('../../Dataset/CandC-test'+str(i)+'.csv', sep=',',header=None)
		X_test = df.values
		df=pd.read_csv('../../Dataset/CandC-train'+str(i)+'.csv', sep=',',header=None)
		X_train = df.values
		X_train = np.array(X_train)
		X_test = np.array(X_test)
		#separating regression O/P parameter as y
		Y_train = X_train[:,-1:]
		X_train = X_train[:,:-1]
		Y_test = X_test[:,-1:]
		X_test = X_test[:,:-1]
		#ridge regression fitting
		regr = linear_model.Ridge (alpha = j)
		regr.fit(X_train, Y_train)
		#building coefficient array
		W = np.append(np.array([regr.intercept_]), np.array(regr.coef_).T, axis=0)
		#exporting coefficient array as csv file
		my_df = pd.DataFrame(W)
		my_df.to_csv('coeff_q6_'+str(i)+'.csv', index=False, header=False)
		Y_pred = regr.predict(X_test)
		#computing rss
		rss = ((Y_pred - Y_test)**2).sum()
		a.append(rss)
	b.append(a)	
#exporting rss values as csv file	
my_df = pd.DataFrame(b)
my_df.to_csv('rss.csv', index=False, header=False)	
print "Coefficients have been turned in, in a csv file"



#Mean square error turned out to be same, write why in report