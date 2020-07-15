import math as m
import pandas as pd
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

def euclidean_distances(X_train, sample):
    dict = {}
    for i in range(len(X_train)):
        dis = 0
        for j in range(num_pixels):
            dis+=(X_train[i][j] - sample[j])**2
        dis=dis**(1/2)
        dict[i]=dis
    return dict



(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_pixels = X_train.shape[1] * X_train.shape[2]
print("len test " ,len(X_test))


# Reduced Training data from 60000 to 6000
X_train = X_train[0:10000]
y_train = y_train[0:10000]

# ........................

plt.subplot(221)
input_num=6
predict_number=X_test[input_num]
# Number which is going to be predicted ( X_test[1] )
plt.imshow(predict_number, cmap=plt.get_cmap('gray'))
plt.show()
new_X_train=X_train
new_y_train=y_train
# Resizing 2-D array into 1-D array
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')

# NORMALISATION to 0 - 1
X_train = X_train / 255
X_test = X_test / 255
#................

# label of input
input_label = y_test[input_num]
print("label of input ", input_label)

# Calculating Euclidean distance
dic = euclidean_distances(X_train,X_test[input_num])

l=sorted(dic.items(), key = lambda X:X[1])

correct=0
plt.subplot(221)
while True:
    K = int(input("Enter K and for Exit enter 0 ...."))
    if K==0:
        break
    for i in range(K):
        key = l[i][0]
        if y_train[key]== input_label:
            correct+=1
            print("predicted output ", y_train[key])
        else:
            print("predicted wrong output ", y_train[key])
    accuracy = correct/K
    print(accuracy)
    correct=0