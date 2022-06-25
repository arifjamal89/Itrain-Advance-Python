#SVM
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
from sklearn import svm
#from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
clf = svm.SVC()
clf.fit(X_train,Y_train)
# predict the response
pred = clf.predict(X_test)
# Step 4: # evaluate accuracy for classifier
print ("SVC accuracy score : ",accuracy_score(Y_test, pred))

# Step 5: Save the Model
from sklearn.externals import joblib
joblib.dump(clf, 'mymodel.pkl')
# Step 6: Load the Model & Prediction
clf = joblib.load('mymodel.pkl')
clf



#VISUALIZATION

from yellowbrick.classifier import ClassificationReport
# Instantiate the classification model and visualizer
visualizer = ClassificationReport(clf, classes=[0,1,2])
visualizer.fit(X_train, Y_train) # Fit the training data to the visualizer
visualizer.score(X_test, Y_test) # Evaluate the model on the test data
g = visualizer.poof() # Draw/show/poof the data
