#importing modules
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pandas as pd

#reading training data
df=pd.read_csv('../../Dataset/DS1-train.csv', sep=',',header=None)
X = df.values
X = np.array(X)
#separating indicator variables
Y = X[:,20:]
X = X[:,:20]
Y_train = []
Y = Y.tolist()
#reducing to one column of indicator variables, s.t. 0 if it belongs to Class 1 and 1 if it belongs to Class 1
for x in xrange(2800):
	Y_train.append(Y[x].index(1))
#k-NN objects
neigh1 = KNeighborsClassifier(n_neighbors=1)
neigh2 = KNeighborsClassifier(n_neighbors=2)
neigh3 = KNeighborsClassifier(n_neighbors=3)
neigh4 = KNeighborsClassifier(n_neighbors=4)
#fitting with k-NN objects
neigh1.fit(X, Y_train)
neigh2.fit(X, Y_train)
neigh3.fit(X, Y_train)
neigh4.fit(X, Y_train) 
#reading test data
df=pd.read_csv('../../Dataset/DS1-test.csv', sep=',',header=None)
X_test = df.values
X_test = np.array(X_test)
#separating indicator variables
Y_test = X_test[:,20:]
X_test = X_test[:,:20]

Y_true = []
Y_test = Y_test.tolist()
#reducing to one column of indicator variables, s.t. 0 if it belongs to Class 1 and 1 if it belongs to Class 1
for x in xrange(1200):
	Y_true.append(Y_test[x].index(1))

Y_predicted = []	
#prdicting with k-NN objects
Y_predicted.append(neigh1.predict(X_test))
Y_predicted.append(neigh2.predict(X_test))
Y_predicted.append(neigh3.predict(X_test))
Y_predicted.append(neigh4.predict(X_test))

Y_predicted = np.array(Y_predicted)
Y_true = np.array(Y_true)

accuracy = []
precision = []
recall = []
F = []
#finding best fit measures
print "NN\tAccuracy Precision Recall F-measure\n"
for i in xrange(4):
	accuracy.append(100.0*accuracy_score(Y_true, Y_predicted[i]))
	precision.append(100.0*precision_score(Y_true, Y_predicted[i], average='binary'))
	recall.append(100.0*recall_score(Y_true, Y_predicted[i], average='binary'))
	F.append(100.0*f1_score(Y_true, Y_predicted[i], average='binary'))
	print "%d\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\n" %(i+1, accuracy[i], precision[i], recall[i], F[i])
#best fit measures in csv file	
res= np.column_stack((accuracy,precision,recall,F))	
my_df = pd.DataFrame(res)
my_df.to_csv('best_fit_measures.csv', index=False, header=False)

print "Best fit measure have been turned in best_fit_measures.csv file"
#performs best for nn =1

