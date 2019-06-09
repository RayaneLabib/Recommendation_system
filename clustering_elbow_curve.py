from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import time
import matplotlib.pyplot as plt

NB_CLUSTERS = 40

def readFromCSV(file, delimiter = ";"):
	csvFile = open(file,"r",encoding='UTF')
	line = csvFile.readline()
	tmp = line.split(delimiter)
	matrix = np.array([[int(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4]), float(tmp[5]), float(tmp[6]), int(tmp[7])]])
	counter = 1
	somme = 0
	tmp_time = start
	for line in csvFile:
		counter += 1
		tmp_time = time.time()- tmp_time
		somme += tmp_time
		print('line number :', counter, ' & time:', time.time() - start, end="\r")
		tmp = line.split(delimiter)
		#print(tmp[1],tmp[2],tmp[3],tmp[4],tmp[7])
		matrix = np.concatenate((matrix, np.array([[int(tmp[1]), int(tmp[2]), float(tmp[3]), float(tmp[4]), float(tmp[5]), float(tmp[6]), int(float(tmp[7]))]])))
	csvFile.close()
	return matrix

def getPropertyType(type):
	dict = {"['HOUSE']":0,\
			"['APARTMENT']":1}#,
			#"['GROUND']":2,\
			#"['INVESTMENT_PROPERTY']":3,\
			#"['COMMERCIAL']":4,\
			#"['OTHER']":5,\
			#"['INDUSTRIES']":6,\
			#"['CAR_PLACES']":7,\
			#"['OFFICES']":8,\
			#"['ACQUISITIONS']":9}
	if type in dict:
		return dict[type]
	return -1

def getTransferType(type):
	dict = {"['SALE']":0,\
			"['RENT']":1}
	if type in dict:
		return dict[type]
	return -1



if __name__ == '__main__':
	start = time.time()
	X = readFromCSV('searchCriterion.csv', delimiter=';')
	scaler = MinMaxScaler()
	X[:,[2,3,6]] = scaler.fit_transform(X[:,[2,3,6]])
	X[:,[4,5]] = X[:,[4,5]]/100
	dist = np.zeros(NB_CLUSTERS)
	#K-means application
	for i in range(1,NB_CLUSTERS):
		print('time before kmeans algorithm for k =',i,':', time.time() - start, end="\n")
		kmeans_model = KMeans(n_clusters=i, random_state=0).fit(X)
		kmeans  = kmeans_model.predict(X)
		print(kmeans_model.n_iter_)
		print('time before kmeans algorithm for k =',i,':', time.time() - start, end="\n")
		kmean = np.array(kmeans)
		print(kmean)
		for j in range(0,i):
			dist[i-1] += np.sum(euclidean_distances(X[kmean==j][:], [kmeans_model.cluster_centers_[j]], squared=True))
	plt.plot(dist,'o-')
	plt.show()
	
	# X = X[kmean==1][:] #k optimal k = 10
	dist_intra = np.zeros(NB_CLUSTERS)
	dist_inter = np.zeros(NB_CLUSTERS)
	#K-means application
	for i in range(1,NB_CLUSTERS):
		print('time before kmeans algorithm for k =',i,':', time.time() - start, end="\n")
		kmeans_model = KMeans(n_clusters=i, random_state=0).fit(X)
		kmeans  = kmeans_model.predict(X)
		print('time before kmeans algorithm for k =',i,':', time.time() - start, end="\n")
		kmean = np.array(kmeans)
		for j in range(0,i):
			dist_intra[i-1] += np.sum(euclidean_distances(X[kmean==j][:], [kmeans_model.cluster_centers_[j]], squared=True))
			dist_inter[i-1] += np.sum(euclidean_distances(kmeans_model.cluster_centers_, [kmeans_model.cluster_centers_[j]], squared=True))

	plt.plot(dist_intra,'o-')
	plt.plot(dist_inter,'ro-')
	
	clustered_results = np.concatenate((X,np.array([kmeans]).T),axis=1)
	tmp = np.argsort(clustered_results[:,2])
	tmp = clustered_results[tmp,:]
	open("clustering-kmeans.csv","w").close()
	f = open("clustering-kmeans.csv","a")
	for line in tmp:
		for el in line:
			f.write(str(el)+";")
		f.write("\n")
	f.close()
	
	# # DBSCAN application
	# print('time before dbscan algorithm:', time.time() - start, end="\n")
	# dbscan_clusters = DBSCAN(eps=0.1, min_samples=100, n_jobs=-1).fit_predict(X)
	# print('time after dbscan algorithm:', time.time() - start, end="\n")
	# clustered_results = np.concatenate((X,np.array([dbscan_clusters]).T),axis=1)
	# tmp = np.argsort(clustered_results[:,2])
	# tmp = clustered_results[tmp,:]
	# open("clustering-dbscan.csv","w").close()
	# f = open("clustering-dbscan.csv","a")
	# for line in tmp:
		# for el in line:
			# f.write(str(el)+";")
		# f.write("\n")
	# f.close()

	# # Hierarchical clustering application
	# print('time before hierarchical clustering algorithm:', time.time() - start, end="\n")
	# hierarchical_clusters = AgglomerativeClustering(n_clusters=20).fit_predict(X)
	# print('time after hierarchical clustering algorithm:', time.time() - start, end="\n")
	# clustered_results = np.concatenate((X,np.array([hierarchical_clusters]).T),axis=1)
	# tmp = np.argsort(clustered_results[:,2])
	# tmp = clustered_results[tmp,:]
	# open("clustering-hierarchical.csv","w").close()
	# f = open("clustering-hierarchical.csv","a")
	# for line in tmp:
		# for el in line:
			# f.write(str(el)+";")
		# f.write("\n")			
	# f.close()
