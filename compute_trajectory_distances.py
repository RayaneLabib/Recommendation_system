import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
import time
import numba
import random

LEARNING_RATE = 0.03
REGULARISATION_RATE = 0.02
NB_FEATURES = 10

def make_trajectory_matrix():
    counter = 0
    properties_dict = {}

    nb_missing_goods = 0
    nb_found_goods = 0
    #extraction of all properties IDs and assignement of an index value to each of them
    file  = open("properties.csv")
    file.readline()
    for line in file:
        properties_dict[line.split(";")[0]] = counter
        counter += 1
    file.close()


    #storage of all the lines of the file containing an IP address and all the IDs of the checked properties
    properties_per_ip = []

    file  = open("stepByStep1.csv")
    for line in file:
        properties_per_ip.append(line)
    file.close()

    #creation of the ip/checked properties matrix
    nb_ip = len(properties_per_ip)
    nb_properties = counter

    #extraction of all IPs and assignement of an index value to each of them
    ip_dict = {}
    
    data = []
    row = []
    col = []
    
    nb_ip_kept = 0
    nb_to_remove = 0
    for i in range(0,nb_ip):
        to_remove = []
        split_ip_properties = properties_per_ip[i].split(";")
        ip_dict[split_ip_properties[0]] = i - nb_to_remove
        for j in range(1,len(split_ip_properties)-1):
            if split_ip_properties[j] not in properties_dict:
                to_remove.append(split_ip_properties[j])
                nb_missing_goods += 1
            else:
                nb_found_goods += 1
        split_ip_properties.remove("\n")
        for element in to_remove:
            split_ip_properties.remove(element)
        if(len(split_ip_properties) > 1):
            nb_ip_kept += 1
            for j in range(1,len(split_ip_properties)):
                data.append(1)
                row.append(i-nb_to_remove)
                col.append(properties_dict[split_ip_properties[j]])
        else:
            nb_to_remove += 1
    visited_properties_matrix = csr_matrix((data,(row,col)),shape=(nb_ip_kept,nb_properties))

    return visited_properties_matrix, nb_ip_kept, nb_properties

if __name__ == "__main__":  
    visited_properties_matrix, nb_ip_kept, nb_properties = make_trajectory_matrix()
    non_zero_cells_ids = visited_properties_matrix.nonzero()

    test_set_sample = random.sample(range(1,len(non_zero_cells_ids[0])),int(len(non_zero_cells_ids[0])*0.15))
    train_set_id =[[],[]]
    test_set_id =[[],[]]
    counter = 0
    for i in range(len(non_zero_cells_ids[0])):
        if i in test_set_sample:
            test_set_id[0].append(non_zero_cells_ids[0][i])
            test_set_id[1].append(non_zero_cells_ids[1][i])
            counter += 1
        else:
            train_set_id[0].append(non_zero_cells_ids[0][i])
            train_set_id[1].append(non_zero_cells_ids[1][i])            
    
    print(counter,"/",int(len(non_zero_cells_ids[0])*0.15))

    print("learning_rate :",LEARNING_RATE)
    print("regularisation_rate :",REGULARISATION_RATE)
    P = np.random.rand(nb_ip_kept,NB_FEATURES)
    Q = np.random.rand(NB_FEATURES,nb_properties)
    
    error = 0
    for k in range(len(train_set_id[0])):
        i = train_set_id[0][k]
        j = train_set_id[1][k]
        val = np.inner(P[i,:],Q[:,j])
        error += (visited_properties_matrix[i,j] - val)**2 #+ REGULARISATION_RATE * (np.linalg.norm(P[i,:])**2 + np.linalg.norm(Q[:,j])**2)
    print("error term :",error)

    test_set_error = 0
    for k in range(len(test_set_id[0])):
        i = test_set_id[0][k]
        j = test_set_id[1][k]
        val = np.inner(P[i,:],Q[:,j])
        test_set_error += (visited_properties_matrix[i,j] - val)**2 #+ REGULARISATION_RATE * (np.linalg.norm(P[i,:])**2 + np.linalg.norm(Q[:,j])**2)
    print("test set error term :",test_set_error)
    
    start = time.time()
    nb_iter = 1
    while error > 1 and nb_iter <= 1000:
        if nb_iter%30 == 0:
            LEARNING_RATE /= 2
        print()
        print("nb iteration : ",nb_iter)
        print("learning_rate :",LEARNING_RATE)
        print("regularisation_rate :",REGULARISATION_RATE)
        print("nb features :", NB_FEATURES)
        nb_iter += 1
        error = 0
        regularisation_term = 0
        test_set_error = 0
        nb_correct_result = 0
        grad_P = np.zeros((nb_ip_kept,NB_FEATURES))
        grad_Q = np.zeros((NB_FEATURES,nb_properties))

        @numba.jit(nopython=True)
        def compute(non_zero_cells_ids, non_zero_cells_ids_len, NB_FEATURES, LEARNING_RATE, visited_properties_matrix, REGULARISATION_RATE, P, Q, norm_P, norm_Q,grad_P,grad_Q):
            """
            for l in range(len(non_zero_cells_ids[0])):
                print(l,"/",len(non_zero_cells_ids[0]),end="\r")
                i = non_zero_cells_ids[0][l]
                j = non_zero_cells_ids[1][l]
                val = np.inner(P[i,:],Q[:,j])
                for k in range(0,NB_FEATURES):
                    grad_P[i,k] = LEARNING_RATE * (-2 * (visited_properties_matrix[i,j] - val) * Q[k,j] + 2 * REGULARISATION_RATE * (np.linalg.norm(P[i,:])))
                    grad_Q[k,j] = LEARNING_RATE * (-2 * (visited_properties_matrix[i,j] - val) * P[i,k] + 2 * REGULARISATION_RATE * (np.linalg.norm(Q[:,j])))
            return grad_P, grad_Q
            """
            for l in range(non_zero_cells_ids_len):
                # print(l,"/",non_zero_cells_ids_len,end="\r")
                i = non_zero_cells_ids[0][l]
                j = non_zero_cells_ids[1][l]
                val = 0
                for k in range(NB_FEATURES):
                    val += P[i,k] * Q[k,j]
                for k in range(NB_FEATURES):
                    grad_P[i,k] = LEARNING_RATE * (-2 * (visited_properties_matrix[i,j] - val) * Q[k,j] + 2 * REGULARISATION_RATE * (norm_P[i]))
                    grad_Q[k,j] = LEARNING_RATE * (-2 * (visited_properties_matrix[i,j] - val) * P[i,k] + 2 * REGULARISATION_RATE * (norm_Q[j]))
            return grad_P, grad_Q
        
        start = time.time()
        grad_P, grad_Q = compute(tuple(train_set_id), len(train_set_id[0]), NB_FEATURES, LEARNING_RATE, visited_properties_matrix.todense(), REGULARISATION_RATE, P, Q, np.linalg.norm(P,axis=1), np.linalg.norm(Q,axis=0),grad_P,grad_Q)        
        print("Compute time:", time.time() - start)
        """
        for l in range(len(non_zero_cells_ids[0])):
            print(l,"/",len(non_zero_cells_ids[0]),end="\r")
            i = non_zero_cells_ids[0][l]
            j = non_zero_cells_ids[1][l]
            val = np.inner(P[i,:],Q[:,j])
            for k in range(0,NB_FEATURES):
                grad_P[i,k] = LEARNING_RATE * (-2 * (visited_properties_matrix[i,j] - val) * Q[k,j] + 2 * REGULARISATION_RATE * (np.linalg.norm(P[i,:])))
                grad_Q[k,j] = LEARNING_RATE * (-2 * (visited_properties_matrix[i,j] - val) * P[i,k] + 2 * REGULARISATION_RATE * (np.linalg.norm(Q[:,j])))
        """
        P -= grad_P
        Q -= grad_Q

        # print(P)
        # print(grad_P)
        # print(grad_Q)
        for k in range(len(train_set_id[0])):
            i = train_set_id[0][k]
            j = train_set_id[1][k]
            val = np.inner(P[i,:],Q[:,j])
            error += (visited_properties_matrix[i,j] - val)**2# + REGULARISATION_RATE * (np.linalg.norm(P[i,:])**2 + np.linalg.norm(Q[:,j])**2)
            if visited_properties_matrix[i,j] == np.sign(val-0.5):
                nb_correct_result += 1
        print("error term :",error)
        print("percentage of correct values for train set :", nb_correct_result/len(train_set_id[0])*100)

        for i in range(nb_ip_kept):
            regularisation_term += REGULARISATION_RATE * np.linalg.norm(P[i,:])**2 
        
        for j in range(nb_properties):
            regularisation_term += REGULARISATION_RATE * np.linalg.norm(Q[:,j])**2
        print("regularization term :",regularisation_term)

        nb_correct_result = 0
        for k in range(len(test_set_id[0])):
            i = test_set_id[0][k]
            j = test_set_id[1][k]
            val = np.inner(P[i,:],Q[:,j])
            test_set_error += (visited_properties_matrix[i,j] - val)**2 #+ REGULARISATION_RATE * (np.linalg.norm(P[i,:])**2 + np.linalg.norm(Q[:,j])**2)
            if visited_properties_matrix[i,j] == np.sign(val-0.5):
                nb_correct_result += 1
        print("test set error term :",test_set_error)
        print("percentage of correct values for test set :", nb_correct_result/len(test_set_id[0])*100)
        @numba.jit(nopython=True)
        def compute_values(P, Q, nb_ip_kept, nb_properties,NB_FEATURES,visited_properties_matrix):
            nb_zero = 0
            nb_one = 0
            flag_filled = 0
            nb_empty_col = 0
            for b in range(nb_properties):
                for a in range(nb_ip_kept):
                    val =0
                    for k in range(NB_FEATURES):
                        val += P[a,k] * Q[k,b]
                    if val > 0.5:
                        nb_one+=1
                    else:
                        nb_zero+=1
                    if flag_filled == 0 and visited_properties_matrix[a,b] != 0:
                        flag_filled = 1
                if flag_filled == 0:
                    nb_empty_col += 1
                flag_filled = 0
            return nb_zero,nb_one,nb_empty_col
        # nb_zero = 0
        # nb_one = 0
        # for a in range(nb_ip_kept):
        #     for b in range(nb_properties):
        #         val = np.inner(P[a,:],Q[:,b])
        #         if np.sign(val-0.5) == 1:
        #             nb_one+=1
        #         if np.sign(val-0.5) == 0:
        #             nb_zero+=1
        nb_zero,nb_one, nb_empty_col = compute_values(P,Q,nb_ip_kept,nb_properties,NB_FEATURES,visited_properties_matrix.todense())
        print("nb zero :", nb_zero)
        print("nb one :", nb_one)
        print("nb empty col :", nb_empty_col/nb_properties)
    # kmeans_model = KMeans(n_clusters=4, random_state=0, n_jobs=-1).fit(visited_properties_matrix)
    # kmeans = kmeans_model.predict(visited_properties_matrix)

    # print(kmeans)
    # print(nb_found_goods)
    # print(nb_missing_goods)
    # print(nb_ip_kept)
    # print(visited_properties_matrix)
                
    # for key,val in properties_dict.items():
    #     if val == 6827:
    #         print(key)
