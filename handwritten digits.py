from sklearn import datasets
digits = datasets.load_digits()
X = digits.data
y = digits.target
X
y

# Split and Randomize Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
X_train
X_test

# Step 2 Define Classifiers
from sklearn import neighbors
clf = neighbors.KNeighborsClassifier()
from sklearn import svm
clf1 = svm.SVC()
from sklearn import linear_model
clf2 = linear_model.SGDClassifier()
from sklearn import tree
clf3 = tree.DecisionTreeClassifier()
from sklearn.ensemble import RandomForestClassifier
clf4 = RandomForestClassifier()
from sklearn.ensemble import GradientBoostingClassifier
clf5 = GradientBoostingClassifier()

# Step 3 Train the Classifier
clf.fit(X_train,y_train)
clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)
clf3.fit(X_train,y_train)
clf4.fit(X_train,y_train)
clf5.fit(X_train,y_train)




# Step 4: Evaluate the Classifier
print("KNN")
print(clf.score(X_test,y_test))
print("SVM")
print(clf1.score(X_test,y_test))
print("SGD")
print(clf2.score(X_test,y_test))
print("Decision Tree")
print(clf3.score(X_test,y_test))
print("Random Forest")
print(clf4.score(X_test,y_test))
print("Gradient Boosting CLassifier")
print(clf5.score(X_test,y_test))



#COnfusion Matrix
# create a prediction array for our test set
y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

y_pred1 = clf1.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred1)






# Step 5: Save the Model
from sklearn.externals import joblib
joblib.dump(clf, 'mymodel.pkl')
# Step 6: Load the Model & Prediction
clf = joblib.load('mymodel.pkl')
clf