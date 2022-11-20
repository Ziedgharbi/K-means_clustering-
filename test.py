# -*- coding: utf-8 -*-


from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

random.seed(1000)
#### exemple 1 : 2 feature


x,y = make_blobs(n_samples= 200, n_features=2)
x.shape

plt.scatter(x[:,0], x[:,1])


x_train, x_test, y_train, y_test = train_test_split( x,y, test_size=0.2 ) 

# on fixe le nombre de cluster à 2
model= KMeans(n_clusters=2, n_init= 10, max_iter= 300,init='k-means++' )
model.fit(x_train)
model.predict(x_test)
centroid=model.cluster_centers_
centroid.shape

plt.scatter(x_test[:,0], x_test[:,1], c= model.predict(x_test))
plt.scatter(centroid[:,0], centroid[:,1], c='r')

# le score 
inertia=model.inertia_    ## ou bien 
inertia= - model.score(x_train)    


# on réexcute le modèle pour différent nombre de cluster

inertia_m = []
k_range= range(1,11,1)
len(k_range)


for k in k_range:
    model= KMeans(n_clusters=k )
    model.fit(x_train)
    model.predict(x_test)
    inertia_m.append(model.inertia_)
    
    model.predict(x_test)
    centroid=model.cluster_centers_
    
    plt.subplot(int(len(k_range)/3)+1,3, k)
    #plt.tick_params(left=False,bottom=False, labelleft=False, labelbottom=False)
    plt.scatter(x_test[:,0], x_test[:,1], c= model.predict(x_test))
    plt.scatter(centroid[:,0], centroid[:,1], c='r')
    
# elbow methode basée sur le coude :  fonction de cout determine le nombre de clusters
# tracer le score en fonction du nombre de cluster 
plt.plot(k_range,inertia_m)


#######################################################################################
#### exemple 2 : 3 feature
x,y = make_blobs(n_samples= 200, n_features=3)
x.shape



ax = plt.axes(projection='3d')
ax.scatter3D(x[:,0], x[:,1], x[:,2], cmap='Greens');


x_train, x_test, y_train, y_test = train_test_split( x,y, test_size=0.2 ) 

# on fixe le nombre de cluster à 3
model= KMeans(n_clusters=3, n_init= 10, max_iter= 300,init='k-means++' )
model.fit(x_train)
model.predict(x_test)
centroid=model.cluster_centers_
centroid.shape

ax = plt.axes(projection='3d')
ax.scatter3D(x_test[:,0], x_test[:,1],x_test[:,2], c= model.predict(x_test))
ax.scatter3D(centroid[:,0], centroid[:,1],centroid[:,2], c='r')


# le score 
inertia=model.inertia_    ## ou bien 
inertia= - model.score(x_train)    


# on réexcute le modèle pour différent nombre de cluster

inertia_m = []
k_range= range(1,11,1)
len(k_range)

fig = plt.figure()

for k in k_range:
    model= KMeans(n_clusters=k )
    model.fit(x_train)
    model.predict(x_test)
    inertia_m.append(model.inertia_)
    
    model.predict(x_test)
    centroid=model.cluster_centers_
    
    ax = fig.add_subplot(int(len(k_range)/3)+1,3, k, projection='3d')
    ax.tick_params(left=False,bottom=False, labelleft=False, labelbottom=False)
    ax.scatter3D(x_test[:,0], x_test[:,1],x_test[:,2], c= model.predict(x_test))
    ax.scatter3D(centroid[:,0], centroid[:,1],centroid[:,2], c='r')

    
# elbow methode basée sur le coude :  fonction de cout determine le nombre de clusters
# tracer le score en fonction du nombre de cluster 
plt.plot(k_range,inertia_m)
