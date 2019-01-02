#Importing modules
import csv
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#To read required data from file and store in array
def read_data(path):
	f = open ( path , 'r')

	
	line = f.readline()
	while line.startswith("%"):
		last_pos = f.tell()
		line = f.readline()
	f.seek(last_pos)
	line = f.read().splitlines()

	size = line[0].split()
	
	line = line[1:]
	i1 = int(size[0])
	j1 = int(size[1])
	
	
	k=0
	arr = np.zeros((i1,j1))
	for j  in xrange(j1):
	        for i in xrange(i1):
	            arr[i][j]=int(line[k])
	            k+=1 
	f.close()                       

	return arr
#storing data required for this question in an array	
X_train = read_data('../../Data_LR(DS2)/data_students/Train_features')

X_test = read_data('../../Data_LR(DS2)/data_students/Test_features')
Y_train = read_data('../../Data_LR(DS2)/data_students/Train_labels')[:,0]
Y_test = read_data('../../Data_LR(DS2)/data_students/Test_labels')[:,0]

#Storing data required for q8 in csv file
X_train_q8 = read_data('../../Data_LR(DS2)/data_students/Train_q8_features')

X_test_q8 = read_data('../../Data_LR(DS2)/data_students/Test_q8_features')
Y_train_q8 = read_data('../../Data_LR(DS2)/data_students/Train_q8_labels')
Y_test_q8 = read_data('../../Data_LR(DS2)/data_students/Test_q8_labels')

T_train_q8 = np.append(np.array(X_train_q8),np.array(Y_train_q8),axis=1)
T_test_q8 = np.append(np.array(X_test_q8),np.array(Y_test_q8),axis=1)
my_df = pd.DataFrame(T_train_q8)
my_df.to_csv('../../Dataset/DS2-train.csv', index=False, header=False)
my_df = pd.DataFrame(T_test_q8)
my_df.to_csv('../../Dataset/DS2-test.csv', index=False, header=False)

#Unregularised Logistic Regressino
lr = LogisticRegression()
lr.fit(X_train, Y_train)

W = np.append(np.array([lr.intercept_]), np.array(lr.coef_).T, axis=0)
#turning in coefficients in a csv file
my_df = pd.DataFrame(W)
my_df.to_csv('coeffs.csv', index=False, header=False)

Y_predict = lr.predict(X_test)


#Regularised Logistic Regression
os.system('cd ../../l1_logreg-0.8.2-i686-pc-linux-gnu/ && ./l1_logreg_train -s Train_features Train_labels 0.01 model')
os.system('cd ../../l1_logreg-0.8.2-i686-pc-linux-gnu/ && ./l1_logreg_classify model Test_features result')

Y_predict_reg = read_data('../../l1_logreg-0.8.2-i686-pc-linux-gnu/result')[:,0]

target_names = ['class 0', 'class 1']
print "No. of iterations for unregularised Logistic Regression: %d\n" %(lr.n_iter_)
print "Performance measures for Unregularised Logistic Regression\n"
print(classification_report(Y_test, Y_predict, target_names=target_names))
print "Performance measures for Regularised Logistic Regression\n"
print(classification_report(Y_test, Y_predict_reg, target_names=target_names))





