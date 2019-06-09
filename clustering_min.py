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

def readFromCSV(file, delimiter=";"):
    csvFile = open(file, "r", encoding='UTF')
    line = csvFile.readline()
    tmp = line.split(delimiter)
    matrix = np.array([[int(tmp[1]), int(tmp[2]), int(tmp[3]), int(
        tmp[4]), float(tmp[5]), float(tmp[6]), int(tmp[7])]])
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
        matrix = np.concatenate((matrix, np.array([[int(tmp[1]), int(tmp[2]), float(
           tmp[3]), float(tmp[4]), float(tmp[5]), float(tmp[6]), int(float(tmp[7]))]])))
        # matrix = np.concatenate((matrix, np.array([[int(tmp[1]), int(tmp[2]), float(
        #     tmp[3]), float(tmp[4]), float(tmp[5]), int(float(tmp[6]))]])))
    csvFile.close()
    return matrix
def getPropertyType(type):
    dict = {"['HOUSE']": 0,
            "['APARTMENT']": 1}
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
    tmp[:, [2, 3, 6]] = scaler.inverse_transform(tmp[:, [2, 3, 6]])
    open(filename, "w").close()
    f = open(filename, "a")
    for line in tmp:
        for el in line:
            f.write(str(float(el))+";")
        f.write("\n")
    f.close()


if __name__ == '__main__':
    start = time.time()
    X = readFromCSV('searchCriterion.csv', delimiter=';')
    scaler = MinMaxScaler()
    X[:, [2, 3, 6]] = scaler.fit_transform(X[:, [2, 3, 6]])
    X[:, [4, 5]] = X[:, [4, 5]]/100
    # K-means application
    print('time before kmeans algorithm :', time.time() - start, end="\n")
    kmeans_model = KMeans(n_clusters=4, random_state=0, n_jobs=-1).fit(X)
    kmeans = kmeans_model.predict(X)
    kmean = np.array(kmeans)
    # visualizeData(X,kmean,'kmeans')
    print('time before kmeans algorithm :', time.time() - start, end="\n")

    outputLabels(X,kmeans,"separation.csv",scaler)

    X[:, [2, 3, 6]] = scaler.inverse_transform(X[:, [2, 3, 6]])

    scaler = StandardScaler()
    X[:, [2, 3, 6]] = scaler.fit_transform(X[:, [2, 3, 6]])

    X = X[kmean == 0][:]   
    # print(X)
    #K-means application on case house/buye
    print(len(X))
    dist_intra = np.zeros(NB_CLUSTERS)
    dist_inter = np.zeros(NB_CLUSTERS)
    #K-means application
    previous = 16000
    for i in range(2,NB_CLUSTERS+1):
        print('time before kmeans algorithm for k =',i,':', time.time() - start, end="\n")
        kmeans_model = KMeans(n_clusters=i, random_state=0).fit(X)
        kmeans  = kmeans_model.predict(X)
        print('time before kmeans algorithm for k =',i,':', time.time() - start, end="\n")
        kmean = np.array(kmeans)
        for j in range(0,i):
            dist_intra[i-1] += np.sum(euclidean_distances(X[kmean==j][:], [kmeans_model.cluster_centers_[j]], squared=True))
            dist_inter[i-1] += np.sum(euclidean_distances(kmeans_model.cluster_centers_, [kmeans_model.cluster_centers_[j]], squared=True))
        tmp = dist_intra[i-1]
        print(tmp)
        print(previous)
        # print("Silouette for "+str(i)+" clusters : "+str(np.average(silhouette_samples(X, kmeans))))
        #dist_intra[i-1] = tmp/previous
        print(dist_intra[i-1])
        previous = tmp


    xi = [i for i in range(1, NB_CLUSTERS+1)]
    plt.plot(xi, dist_intra,'o-')
    plt.show()
    plt.plot(xi,dist_inter,'ro-')
    plt.show()

    print('time before kmeans algorithm :', time.time() - start, end="\n")
    kmeans_model = KMeans(n_clusters=8, random_state=0).fit(X)
    kmeans = kmeans_model.predict(X)
    print(np.average(silhouette_samples(X, kmeans)))
    print('time before kmeans algorithm :', time.time() - start, end="\n")
    kmean = np.array(kmeans)
   
    outputLabels(X,kmeans,"clustering-buy_house_min.csv",scaler)
    