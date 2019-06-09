from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.metrics import silhouette_samples
import matplotlib.axes
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import random

NB_CLUSTERS = 20

# def readFromCSV(file, delimiter=";"):
#     csvFile = open(file, "r", encoding='UTF')
#     line = csvFile.readline()
#     tmp = line.split(delimiter)
#     matrix = np.array([[int(tmp[1]), int(tmp[2]), int(tmp[3]), int(
#         tmp[4]), float(tmp[5]), float(tmp[6]), int(tmp[7])]])
#     counter = 1
#     tmp_time = start
#     file_content = []
#     for line in csvFile:
#         counter += 1
#         tmp_time = time.time() - tmp_time
#         print('line number :', counter, ' & time:',
#               time.time() - start, end="\r")
#         file_content.append(line)
#     csvFile.close()
#     counter = 0
#     nb_el_file = len(file_content)
#     for tmp in file_content:
#         counter += 1
#         tmp_time = time.time() - tmp_time
#         print('element number :', counter,"/",nb_el_file, ' & time:',
#               time.time() - start, end="\r")
#         tmp = line.split(delimiter)
#         # print(tmp[1],tmp[2],tmp[3],tmp[4],tmp[7])
#         matrix = np.concatenate((matrix, np.array([[int(tmp[1]), int(tmp[2]), float(
#             tmp[3]), float(tmp[4]), float(tmp[5]), float(tmp[6]), int(float(tmp[7]))]])))
#     print(matrix)
#     return matrix

def readFromCSV(file, delimiter=";"):
    csvFile = open(file, "r", encoding='UTF')
    line = csvFile.readline()
    tmp = line.split(delimiter)
    matrix = np.array([[int(tmp[1]), int(tmp[2]), int(tmp[3]),
        float(tmp[4]), float(tmp[5]), int(tmp[6])]])
    #matrix = np.array([[int(tmp[1]), int(tmp[2]), int(tmp[3]), int(
    #    tmp[4]), float(tmp[5]), float(tmp[6]), int(tmp[7])]])
    counter = 1
    somme = 0
    tmp_time = start
    for line in csvFile:
        counter += 1
        tmp_time = time.time() - tmp_time
        somme += tmp_time
        print('line number :', counter, ' & time:',
              time.time() - start, end="\r")
        tmp = line.split(delimiter)
        # print(tmp[1],tmp[2],tmp[3],tmp[4],tmp[7])
        #matrix = np.concatenate((matrix, np.array([[int(tmp[1]), int(tmp[2]), float(
        #    tmp[3]), float(tmp[4]), float(tmp[5]), float(tmp[6]), int(float(tmp[7]))]])))
        matrix = np.concatenate((matrix, np.array([[int(tmp[1]), int(tmp[2]), float(
            tmp[3]), float(tmp[4]), float(tmp[5]), int(float(tmp[6]))]])))
    csvFile.close()
    return matrix
def getPropertyType(type):
    dict = {"['HOUSE']": 0,
            "['APARTMENT']": 1}  # ,
    # "['GROUND']":2,\
    # "['INVESTMENT_PROPERTY']":3,\
    # "['COMMERCIAL']":4,\
    # "['OTHER']":5,\
    # "['INDUSTRIES']":6,\
    # "['CAR_PLACES']":7,\
    # "['OFFICES']":8,\
    # "['ACQUISITIONS']":9}
    if type in dict:
        return dict[type]
    return -1


def getTransferType(type):
    dict = {"['SALE']": 0,
            "['RENT']": 1}
    if type in dict:
        return dict[type]
    return -1


def visualizeData(data, labels, algo_name):
    n = max(labels)

    pca = sklearnPCA(n_components=2)
    data_transformed = pd.DataFrame(pca.fit_transform(data))

    def r(): return random.randint(0, 255)
    for i in range(0, n+1):
        plt.scatter(data_transformed[labels == i][0], data_transformed[labels == i]
                    [1], label='label '+str(i+1), c='#%02X%02X%02X' % (r(), r(), r()))

    plt.title('Visualization of ' + algo_name+' algorithm')
    plt.xlabel('1st component')
    plt.ylabel('2nd component')
    plt.legend()
    plt.show()
";\n"
""

def outputLabels(data, labels, filename,scaler):
    clustered_results = np.concatenate((data, np.array([labels]).T), axis=1)
    tmp = np.argsort(clustered_results[:, len(clustered_results[0])-1])
    tmp = clustered_results[tmp, :]
    tmp[:, [2, 5]] = scaler.inverse_transform(tmp[:, [2, 5]])
    open(filename, "w").close()
    f = open(filename, "a")
    for line in tmp:
        for el in line:
            f.write(str(el)+";")
        f.write("\n")
    f.close()


if __name__ == '__main__':
    start = time.time()
    X = readFromCSV('searchCriterion.csv', delimiter=';')
    scaler = MinMaxScaler()
    X[:, [2, 5]] = scaler.fit_transform(X[:, [2, 5]])
    X[:, [3, 4]] = X[:, [3, 4]]/100
    # K-means application
    print('time before kmeans algorithm :', time.time() - start, end="\n")
    kmeans_model = KMeans(n_clusters=4, random_state=0, n_jobs=-1).fit(X)
    kmeans = kmeans_model.predict(X)
    kmean = np.array(kmeans)
    # visualizeData(X,kmean,'kmeans')
    print('time before kmeans algorithm :', time.time() - start, end="\n")

    outputLabels(X,kmeans,"separation.csv",scaler)

    X[:, [2, 5]] = scaler.inverse_transform(X[:, [2, 5]])

    scaler = StandardScaler()
    X[:, [2, 5]] = scaler.fit_transform(X[:, [2, 5]])

    X = X[kmean == 3][:]   
    print(X)
    #K-means application on case house/buye
    print(len(X))
    dist_intra = np.zeros(NB_CLUSTERS)
    dist_inter = np.zeros(NB_CLUSTERS)
    #K-means application
    previous = 16000
    for i in range(1,NB_CLUSTERS+1):
        print('time before kmeans algorithm for k =',i,':', time.time() - start, end="\n")
        kmeans_model = KMeans(n_clusters=i, random_state=0, n_jobs=-1).fit(X)
        kmeans  = kmeans_model.predict(X)
        print('time before kmeans algorithm for k =',i,':', time.time() - start, end="\n")
        kmean = np.array(kmeans)
        for j in range(0,i):
            dist_intra[i-1] += np.sum(euclidean_distances(X[kmean==j][:], [kmeans_model.cluster_centers_[j]], squared=True))
            dist_inter[i-1] += np.sum(euclidean_distances(kmeans_model.cluster_centers_, [kmeans_model.cluster_centers_[j]], squared=True))
        tmp = dist_intra[i-1]
        print(tmp)
        print(previous)
        #dist_intra[i-1] = tmp/previous
        print(dist_intra[i-1])
        previous = tmp


    xi = [i for i in range(1, NB_CLUSTERS+1)]
    plt.plot(xi, dist_intra,'o-')
    plt.show()
    plt.plot(xi,dist_inter,'ro-')
    plt.show()

    print('time before kmeans algorithm :', time.time() - start, end="\n")
    kmeans_model = KMeans(n_clusters=4, random_state=0).fit(X)
    kmeans = kmeans_model.predict(X)
    print(np.average(silhouette_samples(X, kmeans)))
    print('time before kmeans algorithm :', time.time() - start, end="\n")
    kmean = np.array(kmeans)
   
    outputLabels(X,kmeans,"clustering-location_house3.csv",scaler)
    
    #visualizeData(X, kmean, 'kmeans')

    # # DBSCAN application
    # print('time before dbscan algorithm:', time.time() - start, end="\n")
    # dbscan_clusters = DBSCAN(eps=0.01, min_samples=50, n_jobs=-1).fit_predict(X)
    # print('time after dbscan algorithm:', time.time() - start, end="\n")
    # dbscan_labels = np.array(dbscan_clusters)

    # outputLabels(X,dbscan_clusters,"clustering-dbscan.csv",scaler)
    
    # #visualizeData(X,dbscan_labels,'DBSCAN')


    # # Hierarchical clustering application
    # print('time before hierarchical clustering algorithm:', time.time() - start, end="\n")
    # hierarchical_clusters = AgglomerativeClustering(n_clusters=7, linkage='single').fit_predict(X)
    # print('time after hierarchical clustering algorithm:', time.time() - start, end="\n")
    # hierarchical_labels = np.array(hierarchical_clusters)


    # outputLabels(X,hierarchical_clusters,"clustering-hierarchical.csv",scaler)

    # #visualizeData(X,hierarchical_labels,'hierarchical clustering')
