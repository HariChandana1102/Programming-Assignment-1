# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 14:22:56 2019

@author: Hari Chandana
"""

import torch
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
    
    
(x_train, y_train, x_test, y_test)=torch.load('mnist.pt')           # Reading the MNIST Dataset
a=torch.reshape(x_train,(1000,784))                                 # Train Dataset
b=torch.reshape(x_test,(100,784))                                 # Test Dataset


"""(x_train, y_train, x_test, y_test)=torch.load('cifar10.pt')     # Reading the CIFAR Dataset
hist=np.zeros((256,3))
tr=np.zeros((1000,768))
sr=np.zeros((100,768))
x=x_train.numpy()
for l in range(1000):                                   # Train Images
    for k in range(3):                                  # RGB Channels
        for i in range(32):                             # Rows of an image
            for j in range(32):                         # Columns of an image
                v=x_train[l][i][j][k]                   # intensity value
                hist[v][k]=hist[v][k]+1                 # Histogram of all three channels
    p=hist.reshape(1,768)                               # Reshaping the histogram
    tr[l]=p

for l in range(100):                                   # similarly for test images
    for k in range(3):
        for i in range(32):
            for j in range(32):
                v=x_test[l][i][j][k]
                hist[v][k]=hist[v][k]+1
    p=hist.reshape(1,768)
    sr[l]=p
a=torch.tensor(tr,dtype=torch.int32) 
b=torch.tensor(sr,dtype=torch.int32)"""


c1=torch.reshape(y_train,(1000,1))              # Train Labels
d1=torch.reshape(y_test,(100,1))                 # Test labels
accuracy=torch.zeros(6)                         # Array initialization for accuracy
confusion=torch.zeros(10,10)                    # Array Initialization for confusion matrix
y=torch.zeros(1000,1)                           # Obtained Test label from training with 1000 samples
c=torch.tensor(c1,dtype=torch.float)            # Datatype conversion
r1=np.zeros((100,1))                            # Obatined Test labels
r=np.array(r1,dtype='uint8')

cnt=np.zeros((100,10))                          # Array initialization of 'cnt' for extraction of highest frequency label
a=torch.tensor(a,dtype=torch.int32)             # Type conversion
b=torch.tensor(b,dtype=torch.int32)             # Type Conversion

for kvalue in range(1,6):                       # For a given k value
    for m in range(100):                        # Test sample
        for n in range(1000):                   # Train smaple
            
            xx=a[n,:]-b[m,:]                    # Computation of euclidean distance
            x1=torch.tensor(xx,dtype=torch.int32)
            x2=x1**2
            y1=torch.sum(x2)
            y[n]=torch.tensor(y1,dtype=torch.int32)
        
        u1=torch.cat([y,c],1)                           # Concatenation of euclidean distances along with their corresponding train labels 
        f,g=torch.sort(u1[:,0], descending=False, out=None)   # Sorting the array in ascending order
        bb=torch.reshape(f,(1000,1))
        s=torch.tensor(bb,dtype=torch.float)
        q=torch.zeros(1000,1)
        for i in range(1000):                           # Sorting the corresponding indices 
                q[i][0]=u1[g[i]][1]
  
        u=torch.cat([s,q],1)                            # Concatenating the euclidean distances and their corresponding labels
    
        for k in range(kvalue):                         # From the sorted array, extract the k number of elements
            tt=u[k][1].numpy()
            tt1=np.array(tt,dtype='uint8')
            cnt[m,tt1]=cnt[m,tt1]+1                     # Increase the corresponding count by one if the label is present
        if np.max(cnt[m,:])==1:                         # If at all the labels are of equal majority,then assign the label which was obtained for k=1 
                r1=u[0][1]   
                r[m]=np.array(r1,dtype='uint8')
        else:
                    r1=np.argmax(cnt[m,:])   
                    r[m]=np.array(r1,dtype='uint8')  
                    c1=c1+1
    rr=torch.tensor(r,dtype=torch.uint8)
    y_true=d1                                            # Groundtruth labels
    y_pred=rr                                            # Obtained labels
    confusion=confusion_matrix(y_true,y_pred)            # Confusion matrix
    accuracy[kvalue]=accuracy_score(y_true,y_pred)*100   # Accuracy measure
    
x_label =[0,1,2,3,4,5]                                  # plotting the accuracy plot
accuracy=accuracy.numpy()
plt.figure()
plt.plot(x_label, accuracy, color='green', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='blue', markersize=12)
plt.xlabel('k value')
plt.ylabel('Accuracy')
plt.title('Accuracy Plot')
plt.show()
    


