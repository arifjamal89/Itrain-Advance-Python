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

# Random FOrest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
clf1 = RandomForestClassifier(n_estimators=100,random_state=42)
# Step 3 Train the Classifier
clf1.fit(X_train,y_train)
pred1 = clf1.predict(X_test)
# Step 4: Evaluate the Classifier
print ("Random Forest accuracy score : ",accuracy_score(y_test, pred1))

# GradientBoosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
clf2 = GradientBoostingClassifier(learning_rate=0.01,random_state=1)
# Step 3 Train the Classifier
clf2.fit(X_train,y_train)
pred2 = clf2.predict(X_test)
# Step 4: Evaluate the Classifier
print ("GBC accuracy score : ",accuracy_score(y_test, pred2))



