#importing modules
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import Imputer

#reading incomplete CandC set
df=pd.read_csv('CandC.csv', sep=',',header=None)
X = df.values
X = np.array(X)

X = X[:,5:] #first five columns removed since they are said to be non-predictive on official website
#replacing '?' with NaN -Not a Number
for i in xrange(1994):
	for j in xrange(122):
		if X[i][j] == '?':
			X[i][j] = 'NaN'
#imputing data
imp = Imputer(missing_values='NaN',strategy="mean",axis=0)
X_new = imp.fit_transform(X)
#storing completed data as csv file
my_df = pd.DataFrame(X_new)
my_df.to_csv('CandC_complete.csv', index=False, header=False)
