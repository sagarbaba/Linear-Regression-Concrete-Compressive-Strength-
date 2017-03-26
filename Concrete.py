# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 18:36:10 2017

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 10:55:08 2017

@author: hp
"""

#import matplotlib
import sklearn
from sklearn import linear_model
from matplotlib import pyplot as plt
import pandas
import numpy
import random
from sklearn.metrics import accuracy_score
###################### Data retrieval ##################

df = pandas.read_csv('F:/D.csv')

data_y = df['Concrete']
del df['Concrete']
data_x = df
print (data_x)
print ("\n")
print (data_y)
print ("\n")
X_test = numpy.array(data_x, dtype=float)
Y_test = numpy.array(data_y, dtype=float)

#print X_test.shape
#print Y_test.shape

lrm = linear_model.LinearRegression()
#lrm = linear_model.LogisticRegression()

lrm.fit(X_test, Y_test)

print (lrm.coef_)

##################### Processing the data ###########

#X_test_t = numpy.transpose(X_test)
#Identity_mat = numpy.identity(5)
'''
Err = []
ErrX = []

f=-13.8
j=0.001
for i in range(0, 100,1):
    t1 = numpy.dot(X_test_t, X_test)
    t2 = (f+j)*Identity_mat
    f=f+j
    t3 = numpy.add(t1,t2)
    t4 = numpy.linalg.inv(t3)
    t5 = numpy.dot(t4,X_test_t)
    a = numpy.dot(t5,Y_test)

    Y_new = numpy.dot(X_test,a)
    ErrorY = numpy.array(numpy.subtract(Y_test,Y_new))
    #print ErrorY
    error = 0
    for k in range(0,ErrorY.size,1):
        if ErrorY[k] >= (-0.1) and ErrorY[k] <= (0.1):
            error=error+1
    sizeOfErrorY = ErrorY.size
    error = float(float(error)/float(sizeOfErrorY))
    error = error*float(100)
    ErrX.append(f-j)
    Err.append(error)

##################   Plotting Error  vs lamda ##############
#print Err
#plt.scatter(ErrX, Err)
#plt.show()
'''
############ Lambda Decision################
'''
lmd = -13.76
t1 = numpy.dot(X_test_t, X_test)
t2 = lmd*Identity_mat
t3 = numpy.add(t1,t2)
t4 = numpy.linalg.inv(t3)
t5 = numpy.dot(t4,X_test_t)
a_best = numpy.dot(t5,Y_test)
Y_new = numpy.dot(X_test,a_best)
print Y_new

wr = [7, 9.5, 4.2 , 6.1, 8]
w = numpy.array(wr)
y = numpy.dot(w,a_best)
print y
'''
#wr=[540,0,0,162,2.5,1040,676,28]
#w=numpy.array(wr)
#yr=lrm.predict(w)
#print(yr)
random.shuffle(X_test) 
train_set, test_set = X_test[:980], X_test[980:]

w=numpy.array(test_set)
yr=lrm.predict(w)
print(yr)



err=yr-Y_test[980:1030]
print(err)
#print accuracy_score(Y_test[980:1030],yr)
c=0
for i in range(0, 50,1):
    if err[i] >= -5 and err[i] <=5:
      c=c+1
print("Number of predictions with error in the range -5 to +5")
print(c)
print("% of Accuracy")
print(c/50*100)
#x1=numpy.array(i for i in range(1,51),dtype=int)
l=[]
for i in range(1,51,1):
    l.append(i)
x1 = numpy.array(l)
plt.scatter(x1,err)
plt.show()
