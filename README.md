# Programming_Assignment-1
Unsupervised Classification using K-Nearest Neighbor Classifier
The code is implementing the Unsupervised classification using KNN Classifier.

There are two given datasets:
(i) MNIST
(ii) CIFAR

The code is basically written for MNIST dataset, whereas the CIFAR dataset code is commented

Code in a Crux!

The training and test samples are read. For each test sample, euclidean distance is calculated between all the train samples.
Among all the euclidean distances, the minimum euclidean distance is considered as the closest match and the corresponding label is assigned to the test sample.
Once all the labels are obtained, accuracy is found using the metrics like accuracy_score and confusion_matrix 
