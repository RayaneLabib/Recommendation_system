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
    # start = time.time()
    # dist_intra = np.zeros(NB_CLUSTERS)
    # dist_inter = np.zeros(NB_CLUSTERS)
    # for i in range(1,NB_CLUSTERS+1):
    #     print('time before kmeans algorithm for k =',i,':', time.time() - start, end="\n")
    #     kmeans_model = KMeans(n_clusters=i, random_state=0, n_jobs=-1).fit(visited_properties_matrix)
    #     kmeans  = kmeans_model.predict(visited_properties_matrix)
    #     print('time after kmeans algorithm for k =',i,':', time.time() - start, end="\n")
    #     kmean = np.array(kmeans)
    #     for j in range(0,i*10):
    #         dist_intra[i-1] += np.sum(euclidean_distances(visited_properties_matrix[kmean==j][:], [kmeans_model.cluster_centers_[j]], squared=True))
            # dist_inter[i-1] += np.sum(euclidean_distances(kmeans_model.cluster_centers_, [kmeans_model.cluster_centers_[j]], squared=True))
        # tmp = dist_intra[i-1]
        # print(tmp)
        #dist_intra[i-1] = tmp/previous
        # print(dist_intra[i-1])
        # previous = tmp
    # print(kmeans)
    # print(visited_properties_matrix)
    # xi = [i*10 for i in range(1, NB_CLUSTERS+1)]
    # plt.plot(xi, dist_intra,'o-')
    # plt.show()
    # plt.plot(xi,dist_inter,'ro-')
    # plt.show()
    # visited_properties_matrix = visited_properties_matrix.tocsr()
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
    # print(visited_properties_matrix)
    for i in range(51):
        print(i)
        X = visited_properties_matrix[kmean == i][:]
        col_id,properties_id = X.nonzero()
        for j in properties_id:
            for k in range(len(properties_characteristics[j])):
                clustering_file.write(str(properties_characteristics[j][k])+";")
            clustering_file.write(str(i)+"\n")
        # print(properties_id)

    clustering_file.close()
    
    # print(properties_characteristics)
