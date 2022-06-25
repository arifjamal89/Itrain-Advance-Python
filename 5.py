#Random Forest Classifier
#import sklearn as sk

from sklearn import datasets
#Load Data
iris = datasets.load_iris()
X = iris.data
y = iris.target
X
y

# Split and Randomize Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
X_train
X_test

# Step 2 Define Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
clf1 = RandomForestClassifier()
# Step 3 Train the Classifier
clf1.fit(X_train,y_train)
pred1 = clf1.predict(X_test)
# Step 4: Evaluate the Classifier
print ("Random Forest accuracy score : ",accuracy_score(y_test, pred1))


# Step 5: Save the Model
from sklearn.externals import joblib
joblib.dump(clf, 'mymodel.pkl')
# Step 6: Load the Model & Prediction
clf = joblib.load('mymodel.pkl')
clf


