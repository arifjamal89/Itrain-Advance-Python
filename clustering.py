import matplotlib.pyplot as plt
from sklearn import datasets
#Load Data
iris = datasets.load_iris()
X = iris.data
y = iris.target
# Step 1 Model
from sklearn import cluster
cluster = cluster.KMeans(n_clusters=4)
# Step 2 Training
cluster.fit(X)
# Step 3 Evaluation
plt.scatter(X[:,0],X[:,1],c=cluster.labels_)



#Mean Shift Clustering
from sklearn.cluster import MeanShift
ms = MeanShift()
ms.fit(iris.data)




from sklearn.cluster import AgglomerativeClustering
groups = AgglomerativeClustering(n_clusters=2)
groups .fit_predict(iris.data)

from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit(iris.data)
pca_2d = pca.transform(iris.data)
print(pca_2d.shape)
for i in range(0, pca_2d.shape[0]):
    if ms.labels_[i] == 1:
        c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
    elif ms.labels_[i] == 0:
        c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
#plt.title('Mean shift finds 2 clusters)
plt.show()

#Another Method


import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#the imported dataset does not have the required column names so lets add it
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
irisdata = pd.read_csv(url, names=colnames)
irisdata.head()
irisdata['Class'] = pd.Categorical(irisdata["Class"])
irisdata["Class"] = irisdata["Class"].cat.codes
X = irisdata.values[:, 0:4]
y = irisdata.values[:, 4]
from sklearn.cluster import KMeans
# Number of clusters
kmeans = KMeans(n_clusters=3)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_
from sklearn.metrics import classification_report
target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
print(classification_report(irisdata['Class'],kmeans.labels_,target_names=target_names))
#You can see in the classification report that, 91% of our data was predicted accurately. Thats pretty good for an unsupervised algorithm.




#Elbow plot to choose the number of clusters

cost =[] 
for i in range(1, 11): 
    KM = cluster.KMeans(n_clusters = i, max_iter = 500) 
    KM.fit(X) 
      
    # calculates squared error 
    # for the clustered points 
    cost.append(KM.inertia_)      
  
# plot the cost against K values 
plt.plot(range(1, 11), cost, color ='g', linewidth ='3') 
plt.xlabel("Value of K") 
plt.ylabel("Sqaured Error (Cost)") 
plt.show() # clear the plot 