from mpi4py import MPI
import numpy as np
import random
from array import array
import math
import time
import sys
import pickle as pickle
#import resnet
from torchvision import transforms
import torch
from torchvision.datasets import FashionMNIST
from torchvision import models
#import tensorflow.keras as K
import resnet


#Berrut Encoder
def encoder(X,N):
    [K,H,W]=np.shape(X)
    alpha=np.zeros(K)
    for j in range(K):
        alpha[j]=np.cos(((2*j+1)*np.pi)/(2*K))

    all_z=np.zeros(N)
    for i in range(N):
        all_z[i]=np.cos((i*np.pi)/(N))

    coded_X=np.zeros([N,H,W])
    for n in range(N):
        z=all_z[n]
        den=0
        for j in range(K):
            den = den+(np.power(-1, j)) / (z - alpha[j])
        for i in range(K):
            coded_X[n,]=coded_X[n,]+(((np.power(-1, i)) / (z - alpha[i]))/den)*X[i,]
    return coded_X










def welch_decoder(y, K, N, retuned_points_indices):  # Returns the error locations
    # e=3
    e_max = int(((N - 2 * K)+1) / 2)
    e = e_max
    alpha = np.zeros(K)
    for j in range(K):
        alpha[j] = np.cos(((2 * j + 1) * np.pi) / (2 * K))

    all_z = np.zeros(N)
    for i in range(N):
        all_z[i] = np.cos((i * np.pi) / (N))

    returned_z=all_z[retuned_points_indices]

    # a=np.zeros([K+e,1])
    # b=np.zeros([K+e,1])

    A1 = np.vander(returned_z, N=K + e)
    A2 = np.vander(returned_z, N=K + e )
    A2 = -A2[:, 0:K + e-1]
    A2 = A2 * y
    # print("y dimension:" + str(y.shape))
    # print("A2 dimension:" + str(A2.shape))
    A = np.concatenate([A1, A2], axis=1)
    # print("A dimension:"+ str(A.shape))

    # coeffs=np.zeros([N,1])
    # print("Singularvaluse: "+str(svdvals(A)))
    A_psudo_inv = np.linalg.pinv(A)
    coeffs = np.matmul(A_psudo_inv, y)

    a = coeffs[0:K + e, 0]
    b = coeffs[K + e:2 * (K + e)-1, 0]
    b = np.reshape(b, [len(b), 1])
    bb = np.concatenate([b, np.ones([1, 1])])
    #plambda_evals = np.zeros([N, 1])
    qlambda_evals = np.zeros([N, 1])

    # output=np.zeros([10,1])
    # for i in range(N):
    #     plambda_evals[i]=np.polyval(a,all_z[i])
    #     qlambda_evals[i]=np.polyval(bb,all_z[i])
    #     output[i]=plambda_evals[i]/qlambda_evals[i]

    output = np.zeros([N, 1])
    for i in range(N):
        #plambda_evals[i] = np.polyval(a, all_z[i])
        qlambda_evals[i] = np.polyval(bb, all_z[i])

    # idx = np.argpartition(np.abs(plambda_evals.transpose()), e)# ignore numerator
    # print((idx[:, :e]))

    idx = np.argpartition(np.abs(qlambda_evals.transpose()), e)
    # print((idx[:, :e]))
    # sorted = np.sort(np.abs(qlambda_evals.transpose()))
    # sorted_deiff = np.diff(sorted)
    # ratio = sorted_deiff / sorted[0, :N - 1]
    # e = (np.argmax(ratio)) + 1
    #e = np.minimum(int(e), e_max)
    e=e_max #see what happens in experiments
    output = (np.sort(idx[:, :e]))
    # output_reshaped=np.reshape(output,[1,len(output)])

    # return output.transpose()
    return np.sort(idx[:, :e])


#Berrut Decoder
def decoder(Y,K,N,returned_points_indices):


    ####introducing error happens here
    sigma_error=10
    e_max=int(((N-2*K)+1)/2)
    ############################################### ACTUAL number of errors
    e=e_max
    N,num_of_classes=np.shape(Y)
    erroneous_indices=np.random.permutation(returned_points_indices)[:e]
    #print(erroneous_indices)

    ###ERROR
    # print("Y size"+str(Y.shape))
    #print(returned_points_indices)
    Y[erroneous_indices, :]=Y[erroneous_indices,:]+np.random.normal(0,sigma_error,[e,num_of_classes])

    adversary_indices_matrix=np.zeros([e_max, num_of_classes])
    # print(adversary_indices_matrix.shape)
    for i in range(num_of_classes):
        y=Y[returned_points_indices,i]
        y=np.reshape(y,[len(y),1])
        # print("y is "+str(y.shape))
        # print(welch_decoder(y,K,N,returned_points_indices).shape)
        # print(adversary_indices_matrix[:,i].shape)
        adversary_indices_matrix[:,i]=welch_decoder(y,K,N,returned_points_indices)


    flattened_adversary_indices=(adversary_indices_matrix.flatten())
    flattened_adversary_indices=np.reshape(flattened_adversary_indices,[1,len(flattened_adversary_indices)])
    bin_count=(np.bincount(flattened_adversary_indices[0,:].astype(np.int64)))
    temp=((np.argsort(bin_count)))
    error_locations_predicted=temp[-e_max:]
    #print(error_locations_predicted)


    #### The indices of adversaries are learned at this point. We only need to exclude them from
    # print(error_locations_predicted)
    # print(returned_points_indices)

    for i in range(len(error_locations_predicted)):
        loc=np.where(returned_points_indices==error_locations_predicted[i])
        returned_points_indices=np.delete(returned_points_indices,loc)
    # print(returned_points_indices)

    F=len(returned_points_indices)
    alpha=np.zeros(K)
    for j in range(K):
        alpha[j]=np.cos(((2*j+1)*np.pi)/(2*K))

    z_bar=np.zeros(N)
    for i in range(N):
        z_bar[i]=np.cos((i*np.pi)/(N))

    probs=np.zeros([K,10])
    for digit in range(10):
        for i in range(K):
            z=alpha[i]
            den = 0
            for j in range(F):
                den = den + ((np.power(-1,j))/(z - z_bar[returned_points_indices[j]]))
            for l in range(F):
                probs[i,digit] = probs[i,digit] + ((((np.power(-1, l)) / (z - z_bar[returned_points_indices[l]]))/den)*Y[returned_points_indices[l],digit])

    return probs




# #Berrut Decoder without error
# def decoder(Y,K,N,returned_points_indices):
#     F=len(returned_points_indices)
#     alpha=np.zeros(K)
#     for j in range(K):
#         alpha[j]=np.cos(((2*j+1)*np.pi)/(2*K))
#
#     z_bar=np.zeros(N)
#     for i in range(N):
#         z_bar[i]=np.cos((i*np.pi)/(N-1))
#
#     probs=np.zeros([K,num_of_classes])
#     for digit in range(num_of_classes):
#         for i in range(K):
#             z=alpha[i]
#             den = 0
#             for j in range(F):
#                 den = den + ((np.power(-1,j))/(z - z_bar[returned_points_indices[j]]))
#             for l in range(F):
#                 probs[i,digit] = probs[i,digit] + ((((np.power(-1, l)) / (z - z_bar[returned_points_indices[l]]))/den)*Y[returned_points_indices[l],digit])
#
#     return probs


def model_out(data):
    outputs=np.zeros(num_of_classes)


    torch_sample = torch.from_numpy(data).float()
    torch_sample = torch_sample.unsqueeze(0)
    torch_sample = torch_sample.unsqueeze(0)
    outputs= model(torch_sample).detach().numpy()[0]

        # _, predicts = torch.max(pred, 1)
        # predictions[i]=predicts.numpy()[0]

    return outputs



if __name__ == "__main__":

    iterations=1
    #mode="ParM"
    mode="Approxifer"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()



    master_node_id=0
    client_node_id=1
    first_worker_node_id=2

    e_max = 1
    S=0 ## num of stragglers
    N=size-first_worker_node_id
    K=(N-S-2*e_max+1)/2
    K=int(K)
    parity_model_id = first_worker_node_id + N - 1

    background_teraffic_exponent=4
    min_packet_size=10**background_teraffic_exponent
    max_packet_size=2*10**background_teraffic_exponent

    # min_packet_size=1
    # max_packet_size=2

    ## Data set sizes

    ## MNIST
    H=28
    W=28
    num_of_classes=10
    sample_batch = np.zeros([H, W])

    # model=resnet.ResNet18()
    # PATH='../base_model_trained_files/fashion-mnist/resnet18/model.t7'
    # model.load_state_dict(torch.load(PATH))
    #model=models.resnet18(pretrained=True)

    model = resnet.ResNet18()
    PATH = '../base_model_trained_files/fashion-mnist/resnet18/model.t7'
    model.load_state_dict(torch.load(PATH))
    model.eval()
    transform = transforms.Compose([transforms.ToTensor()
            ])



    # if rank==random_packet_sender_id:
    #     dummy_packet = np.random.random(np.random.randint(min_packet_size, max_packet_size))
    #     req=comm.Isend(dummy_packet, dest=random_packet_receiver_id)
    #     req.Wait()
    #
    # if rank==random_packet_receiver_id:
    #     dummy_packet = np.random.random(np.random.randint(min_packet_size, max_packet_size))
    #     req=comm.Irecv(dummy_packet, source=random_packet_receiver_id)
    #     req.Wait()






    wait_times=np.zeros([int(K*iterations),1], dtype=float)

### master nodes
    for itr in range(iterations):
        if rank==master_node_id:


            # print("rank is ", rank)
            # print("Hello from the master node")

            #####  Generating background terafic

            [random_packet_sender_id, random_packet_receiver_id] = np.random.permutation(np.arange(first_worker_node_id, N + first_worker_node_id))[0:2]

            #print(random_packet_sender_id, random_packet_receiver_id)

            for i in range(first_worker_node_id, N+first_worker_node_id):
                comm.send([random_packet_sender_id, random_packet_receiver_id], dest=i, tag=2*itr) # msg 2

            # print("Master sent the ids")


            comm.send("Start", dest=client_node_id, tag=(K+1)*itr+K)#msg 1



            # print("Master loaded datasets")
            data_batch=np.zeros([K,H,W])


##########################################################ParM################################################################
            if mode=="ParM":
                for i in range(K):
                    data_batch[i,]=comm.recv(source=client_node_id, tag=K*itr+i)# msg 3
                    comm.Isend(data_batch[i], dest=first_worker_node_id+i, tag=2*itr+1) #msg 4

                parity=data_batch.sum(axis=0)

                comm.Isend(parity, dest=first_worker_node_id + K, tag=2*itr+1) #msg 4
                # print("Master sent out data")


                receive_objects_array = []
                received_data_from_workers = np.zeros([N, num_of_classes,1])
                for i in range(first_worker_node_id, N + first_worker_node_id):
                    receive_objects_array.append(comm.irecv(received_data_from_workers[i - first_worker_node_id,], source=i, tag=itr)) #msg 5

                # print("Master is waiting for the results . . .")

                #is_received_from_worker = np.array(np.zeros([N, 1]), dtype=bool)
                not_received_yet_ids=[i for i in range(first_worker_node_id,first_worker_node_id+N)]
                parity_is_back=False
                straggler_id=parity_model_id
                while len(not_received_yet_ids)>S:
                    for worker_id in not_received_yet_ids:
                        if receive_objects_array[worker_id-first_worker_node_id].Test():
                            if worker_id==parity_model_id:
                                not_received_yet_ids.remove(worker_id)
                                parity_is_back=True
                            else:
                                label=np.argmax(received_data_from_workers[worker_id - first_worker_node_id,])
                                comm.Isend(label, dest=client_node_id, tag=(K+1)*itr+worker_id-first_worker_node_id)# msg 6
                                not_received_yet_ids.remove(worker_id)

                if parity_is_back:
                    straggler_id=not_received_yet_ids[0]
                    returned_pos_except_parity=[i for i in range(N)]
                    returned_pos_except_parity.remove(straggler_id-first_worker_node_id)
                    returned_pos_except_parity.remove(parity_model_id-first_worker_node_id)
                    received_data_from_workers[straggler_id-first_worker_node_id,]=received_data_from_workers[parity_model_id-first_worker_node_id,]-received_data_from_workers[returned_pos_except_parity,].sum(axis=0)
                    label = np.argmax(received_data_from_workers[straggler_id - first_worker_node_id,])
                    comm.Isend(label, dest=client_node_id, tag=(K+1)*itr+straggler_id - first_worker_node_id)# msg 6

                receive_objects_array[straggler_id - first_worker_node_id].Cancel()

                # print("Job is done")


#####################################################Approxifer########################################################################
            elif mode=="Approxifer":
                for i in range(K):
                    data_batch[i,]=comm.recv(source=client_node_id, tag=K*itr+i)# msg 3

                ## encoding using Berrut
                coded_batch=encoder(data_batch,N)

                for i in range(N):
                    comm.Isend(coded_batch[i], dest=first_worker_node_id+i, tag=2*itr+1) #msg 4


                # print("Master sent out data")


                receive_objects_array = []
                received_data_from_workers = np.zeros([N, num_of_classes])
                for i in range(first_worker_node_id, N + first_worker_node_id):
                    receive_objects_array.append(comm.irecv(received_data_from_workers[i - first_worker_node_id,], source=i, tag=itr)) #msg 5

                # print("Master is waiting for the results . . .")

                #is_received_from_worker = np.array(np.zeros([N, 1]), dtype=bool)
                not_received_yet_ids=[i for i in range(first_worker_node_id,first_worker_node_id+N)]
                enough_for_decoding=False
                while len(not_received_yet_ids)>S:
                    for worker_id in not_received_yet_ids:
                        if receive_objects_array[worker_id-first_worker_node_id].Test():
                            not_received_yet_ids.remove(worker_id)

                if S>0:
                    straggler_id = not_received_yet_ids[0]
                    received_results_ids=np.concatenate((np.arange(0,straggler_id-first_worker_node_id),np.arange(straggler_id-first_worker_node_id+1,N)))
                elif S==0:
                    received_results_ids = np.arange(0, N)

                decoded_result=decoder(received_data_from_workers,K,N,received_results_ids)
                labels=np.argmax(decoded_result,axis=1)

                if S > 0:
                    receive_objects_array[straggler_id - first_worker_node_id].Cancel()

                for i in range(K):
                    comm.Isend(labels[i], dest=client_node_id,tag=(K + 1) * itr + i)  # msg 6












    ####### client

        elif rank==client_node_id:
            message=comm.recv(source=master_node_id,tag=(K+1)*itr+K)## msg 1
            send_times = np.zeros(K)
            recv_times = np.zeros(K)
            # print("Hi from the client node")

            #### sending queries
            for i in range(K):
                comm.send(sample_batch,dest=master_node_id, tag=K*itr+i)# msg 3
                send_times[i]=time.time()


            # print("queries are sent")


            receive_objects_array = []
            for i in range(K):
                receive_objects_array.append(comm.irecv(8, source=master_node_id,tag=(K+1)*itr+i))#msg 6

            receive_times=np.zeros(K)
            ids_whose_labels_not_received_yet = [i for i in range(K)]
            while len(ids_whose_labels_not_received_yet)>0:
                for id in ids_whose_labels_not_received_yet:
                    if receive_objects_array[i].Test():
                        ids_whose_labels_not_received_yet.remove(id)
                        receive_times[id]=time.time()


            # print("labels are received")
            #print(receive_times-send_times)
            #wait_times.append(receive_times-send_times)
            wait_times[np.arange(itr*K,(itr+1)*K,dtype=int)]= np.reshape(receive_times-send_times,[K,1])

            if itr==iterations-1:
                mean=wait_times.mean()
                median=np.percentile(wait_times,50)
                percentile99=np.percentile(wait_times,99)
                percentile995=np.percentile(wait_times,99.5)
                percentile999=np.percentile(wait_times,99.9)
                print(mode)
                print("mean is ", mean)
                print("median is", median)
                print("99 percentile is ", percentile99)
                print("99.5 percentile is", percentile995)
                print("99.9 percentile is ", percentile999)
                stats=[mean, median, percentile995, percentile995, percentile999]
                #stats=str(mean), '  ',str(median),'  ',str(percentile99), '  ',str(percentile995), '  ',str(percentile999)
                np.savetxt('K='+str(K)+'N='+str(N)+'iterations='+str(iterations)+'mode'+str(mode)+'teraficexponent'+str(background_teraffic_exponent)+'e_max'+str(e_max)+"stats.txt", stats)
                np.savetxt('K=' + str(K) + 'N=' + str(N) + 'iterations=' + str(iterations) +'mode'+str(mode)+ 'teraficexponent'+str(background_teraffic_exponent)+'e_max'+str(e_max)+"wait_times.txt", wait_times)





            # preq=comm.Send_init(sample_batch, dest=master_node_id)
            # send_times=np.zeros(K)
            # preq.Start()
            #print((preq.__getattribute__))
            # for i in range(K):
            #     preq.Start()
            #     send_times[i]=time.time()










    ##### worker nodes

        elif rank<size:
            # print("rank is ", rank)


            ##emulating the background teraffic
            ids=np.zeros(2,dtype=int)
            ids=comm.recv(source=0, tag=2*itr) #msg 2

            random_packet_sender_id=ids[0]
            random_packet_receiver_id=ids[1]

            if rank==random_packet_sender_id:
                dummy_packet = np.random.random(np.random.randint(min_packet_size, max_packet_size))
                comm.send(dummy_packet, dest=random_packet_receiver_id, tag=itr)#msg 7

            if rank==random_packet_receiver_id:
                comm.recv(source=random_packet_sender_id, tag=itr)# msg 7





            req=comm.irecv(sample_batch, source=0, tag=2*itr+1)# msg 4
            req.Wait()
            # print("Process", rank, "received the data")
            soft_labels=model_out(sample_batch)
            comm.Isend(soft_labels, dest=0, tag=itr) # msg 5
            # print("Process", rank, "sent the result")


















