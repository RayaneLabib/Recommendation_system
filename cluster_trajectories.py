import compute_trajectory_distances
from sklearn.cluster import KMeans
import time
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, csc_matrix

NB_CLUSTERS = 150

if __name__ == "__main__":
    visited_properties_matrix, nb_ip, nb_properties = compute_trajectory_distances.make_trajectory_matrix()

    kmeans_model = KMeans(n_clusters=51, random_state=0, n_jobs=-1).fit(visited_properties_matrix)
    kmeans = kmeans_model.predict(visited_properties_matrix)
    kmean = np.array(kmeans)

    properties_characteristics = []

    property_file  = open("properties.csv")
    property_file.readline()
    for line in property_file:
        characteristics = line.split(";")
        properties_characteristics.append([1 if characteristics[1]=="APARTMENT" else 0,1 if characteristics[2]=="RENT" else 0,characteristics[3],float(characteristics[4])/100,float(characteristics[5])/100,characteristics[6].replace("\n","")])
    property_file.close()

    open("clustering-trajectories.csv","w").close()
    clustering_file = open("clustering-trajectories.csv","a")
    print(kmean[kmean==50])
    for i in range(51):
        print(i)
        X = visited_properties_matrix[kmean == i][:]
        col_id,properties_id = X.nonzero()
        for j in properties_id:
            for k in range(len(properties_characteristics[j])):
                clustering_file.write(str(properties_characteristics[j][k])+";")
            clustering_file.write(str(i)+"\n")
    clustering_file.close()
    
