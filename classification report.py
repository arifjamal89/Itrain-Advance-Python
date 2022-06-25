#This data sets consists of 3 different types of irisesâ€™ (Setosa, Versicolour, and Virginica) petal and sepal length, stored in a 150x4 numpy.ndarray
#import sklearn as sk

from sklearn import datasets
#Load Data
iris = datasets.load_iris()
X = iris.data
Y = iris.target
X
Y

# Split and Randomize Data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=33)
X_train
X_test

# Step 2 Define Classifier
from sklearn import neighbors
from sklearn.metrics import accuracy_score
clf = neighbors.KNeighborsClassifier()
# Step 3 Train the Classifier
clf.fit(X_train,Y_train)
# predict the response
pred = clf.predict(X_test)
# Step 4: # evaluate accuracy for classifier
print ("KNeighbors accuracy score : ",accuracy_score(Y_test, pred))


# SVM
#from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
clf1 = SVC()
clf1.fit(X_train,Y_train)
# predict the response
pred1 = clf1.predict(X_test)
# Step 4: # evaluate accuracy for classifier
print ("SVC accuracy score : ",accuracy_score(Y_test, pred1))




#Classification Report for KNN
# create a prediction array for our test set
y_pred = clf.predict(X_test)
from sklearn.metrics import classification_report
target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
print(classification_report(Y_test, y_pred,target_names=target_names))


#Classification Report for SVM
# create a prediction array for our test set
y_pred1 = clf1.predict(X_test)
from sklearn.metrics import classification_report
target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
print(classification_report(Y_test, y_pred1,target_names=target_names))
##You can see in the classification report that, 96% of our data was predicted accurately. Thats pretty good for an unsupervised algorithm.