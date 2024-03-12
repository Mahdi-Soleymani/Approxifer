import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import resnet
from torchvision.transforms import transforms
from torchsummary import summary
from torchvision import transforms
import matplotlib.pyplot as plt
import tensorflow.keras as K
from knockknock import telegram_sender
x=1
CHAT_ID: int = 43515446
@telegram_sender(token="1834099231:AAEHY1G5pAGDRXH20vyNuP-WrfUqbc4f-X8", chat_id=CHAT_ID)
def train_your_nicest_model(your_nicest_parameters):
    return {'loss': 0.9} # Optional return value


#### Import data from Keras
#(train_x,train_y), (test_x,test_y)=K.datasets.cifar10.load_data()
(train_x,train_y), (test_x,test_y)=K.datasets.mnist.load_data()
train_x=train_x/255.0
test_x=test_x/255.0

#manual normalization
# mean=[0.5, 0.5, 0.5]
# std=[0.25, 0.25, 0.25]
# mean=[0.4914, 0.4822, 0.4465]
# std=[0.2471, 0.2435, 0.2616]
mean=0
std=1
test_x=(test_x-mean)/std




############ model and data selection   ##########
model=resnet.ResNet18()
PATH='../base_model_trained_files/mnist/resnet18/model.t7'
model.load_state_dict(torch.load(PATH))
model.eval()
transform = transforms.Compose([
        #transforms.Resize(256),
        #transforms.RandomCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        #transforms.Normalize((0.5), (0.5))
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        ])

# dataset = CIFAR10(root='data/', download=True, transform=ToTensor())
# test_dataset = CIFAR10(root='data/', train=False, transform=ToTensor())
#dataset = CIFAR10(root='data/', download=True, transform=transform)

#test_dataset = CIFAR10(root='data/', train=False, transform=transform)
#test_dataset = MNIST(root='data/', train=False, transform=transform,download=True)
test_dataset =MNIST(root='data/', train=False, transform=transform,download=True)
batch_size=128

# train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
# val_loader = DataLoader(val_ds, batch_size, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size, num_workers=0, pin_memory=True)
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




def model_accuracy_pytorch():
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
         correct/total ))
    print(correct, total)


def model_accuracy():
    n = len(test_x)
    predictions = np.zeros([n, 1])
    # trans = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])
    for i in range(n):
        data = test_x[i,]
        torch_sample = torch.from_numpy(data).float()
        # torch_sample = torch_sample.unsqueeze(0)
        #torch_sample = torch_sample.permute(2, 0, 1)
        # torch_sample = trans(torch_sample)
        torch_sample = torch_sample.unsqueeze(0)
        torch_sample = torch_sample.unsqueeze(0)
        pred = model(torch_sample)
        _, predicts = torch.max(pred, 1)
        predictions[i] = predicts.numpy()[0]

    #diff = test_y - predictions
    test_y_reshaped = np.reshape(test_y, [n, 1])
    diff=test_y_reshaped-predictions
    print("Model accuracy is")
    out = (n - np.count_nonzero(diff))/n
    print(out)
    print(n-np.count_nonzero(diff))

    return out


#Berrut Encoder
def encoder(X):
    [NN,H,W]=np.shape(X)



    alpha=parameters.alpha
    # alpha=np.zeros(K)
    # for j in range(K):
    #     alpha[j]=np.cos(((2*j+1)*np.pi)/(2*K))

    all_z=parameters.z_bar
    # all_z=np.zeros(N)
    # for i in range(N):
    #     all_z[i]=np.cos((i*np.pi)/(N))######

    coded_X=np.zeros([parameters.N,H,W])
    for n in range(parameters.N):
        z=all_z[n]
        den=0
        for j in range(parameters.K):
            den = den+(np.power(-1, j)) / (z - alpha[j])
        for i in range(parameters.K):
            coded_X[n,]=coded_X[n,]+(((np.power(-1, i)) / (z - alpha[i]))/den)*X[i,]
    return coded_X


### functions to simplify welch_decoder code
def gt(x):
    out=0
    for i in range(parameters.emax+1):
        out=out+((-1)**i)/(x-parameters.beta[i])
    return out

def gk(x):
    out=0
    for i in range(parameters.K):
        out=out+((-1)**i)/(x-parameters.alpha[i])
    return out

def G(x):
    out=0
    for i in range(parameters.K):
        out=out+(gt(parameters.alpha[i])*(-1)**i)/(x-parameters.alpha[i])
    for i in range(parameters.emax+1):
        out=out+(gk(parameters.beta[i])*(-1)**i)/(x-parameters.beta[i])

    return out

########### Returns null space of a matrix
import scipy
from scipy import linalg, matrix
def null(A, eps=1e-15):
    u, s, vh = scipy.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)

def welch_decoder(y,retuned_points_indices,actual_num_0f_errors):  # Returns the error locations

    alpha=parameters.alpha
    all_z = parameters.z_bar

    returned_z=all_z[retuned_points_indices]


    A1=np.zeros([len(returned_z),parameters.K])
    for i in range(len(returned_z)):
        for j in range(parameters.K):
            A1[i,j]=1/((returned_z[i]-parameters.alpha[j])*G(returned_z[i]))



    A2=np.zeros([len(returned_z),parameters.emax+1])
    for i in range(len(returned_z)):
        for j in range(parameters.emax+1):
            A2[i,j]=1/((returned_z[i]-parameters.beta[j])*G(returned_z[i]))


    A3=np.zeros([len(returned_z),parameters.emax+1])
    for i in range(len(returned_z)):
        for j in range(parameters.emax+1):
            A3[i,j]=(-1)**(j)/((returned_z[i]-parameters.beta[j])*gt(returned_z[i]))


    #print(A3)
    A3y = -A3 * y
    # print("y dimension:" + str(y.shape))
    # print("A2 dimension:" + str(A2.shape))
    A = np.concatenate([A1, A2,A3y], axis=1)

    # A_psudo_inv = np.linalg.pinv(A)
    # coeffs = np.matmul(A_psudo_inv, y)


    uu,ss,vv=np.linalg.svd(A)
    #print(np.matmul(A,np.transpose(vv[-1:,:])))
    null_space=vv[-1:,:]
    # print(np.transpose(null_space))
    # print(A.shape)
    b=np.transpose(null_space[:,-parameters.emax-1:])
    #a = coeffs[0:K + e, 0]
    #b = coeffs[parameters.K + parameters.emax:parameters.K+2*parameters.emax, 0]#### do not forget 1 in the error locator function
    b = np.reshape(b, [len(b), 1])
    #print(A)
    lambda_evals = np.zeros([parameters.N, 1])
    #print(np.linalg.cond(A))

    for i in range(parameters.N):
        for j in range(parameters.emax+1):

            lambda_evals[i] = lambda_evals[i]+(b[j]*(-1)**j)/((all_z[i]-parameters.beta[j])*gt(all_z[i]))

    # idx = np.argpartition(np.abs(plambda_evals.transpose()), e)# ignore numerator
    # print((idx[:, :e]))

    idx = np.argpartition(np.abs(lambda_evals.transpose()), actual_num_0f_errors)
    # print((idx[:, :e]))
    # sorted = np.sort(np.abs(qlambda_evals.transpose()))
    # sorted_deiff = np.diff(sorted)
    # ratio = sorted_deiff / sorted[0, :N - 1]
    # e = (np.argmax(ratio)) + 1
    #e = np.minimum(int(e), e_max)
    #e=e_max #see what happens in experiments
    output = (np.sort(idx[:, :actual_num_0f_errors]))
    # output_reshaped=np.reshape(output,[1,len(output)])

    # return output.transpose()

    return np.sort(idx[:, :actual_num_0f_errors])



#Berrut Decoder
def decoder(Y,returned_points_indices,actual_num_0f_errors):


    ####introducing error happens here
    sigma_error=100

    ############################################### ACTUAL number of errors
    e=parameters.emax

    something,num_of_classes=np.shape(Y)
    erroneous_indices=np.random.permutation(returned_points_indices)[:actual_num_0f_errors]

    # print("actual error locations:")
    # print(erroneous_indices)   #########################################################

    ###ERROR
    # print("Y size"+str(Y.shape))
    #print(returned_points_indices)
    Y[erroneous_indices, :]=Y[erroneous_indices,:]+np.random.normal(0,sigma_error,[actual_num_0f_errors ,num_of_classes])
    #print(Y[erroneous_indices, :])
    adversary_indices_matrix=np.zeros([actual_num_0f_errors, num_of_classes])
    # print(adversary_indices_matrix.shape)
    for i in range(num_of_classes):
        y=Y[returned_points_indices,i]
        y=np.reshape(y,[len(y),1])
        # print("y is "+str(y.shape))
        # print(welch_decoder(y,K,N,returned_points_indices).shape)
        # print(adversary_indices_matrix[:,i].shape)
        adversary_indices_matrix[:,i]=welch_decoder(y,returned_points_indices,actual_num_0f_errors)


    flattened_adversary_indices=(adversary_indices_matrix.flatten())
    flattened_adversary_indices=np.reshape(flattened_adversary_indices,[1,len(flattened_adversary_indices)])
    bin_count=(np.bincount(flattened_adversary_indices[0,:].astype(np.int64)))
    temp=((np.argsort(bin_count)))
    error_locations_predicted=temp[-parameters.emax:]


    # print("predicted error locations:") ########################################
    # print(error_locations_predicted)


    #### The indices of adversaries are learned at this point. We only need to exclude them from
    # print(error_locations_predicted)
    # print(returned_points_indices)

    for i in range(len(error_locations_predicted)):
        loc=np.where(returned_points_indices==error_locations_predicted[i])
        returned_points_indices=np.delete(returned_points_indices,loc)


    #print(returned_points_indices)



    F=len(returned_points_indices)
    alpha=parameters.alpha

    z_bar=parameters.z_bar

    probs=np.zeros([parameters.K,10])
    for digit in range(10):
        for i in range(parameters.K):
            z=alpha[i]
            den = 0
            for j in range(F):
                den = den + ((np.power(-1,j))/(z - z_bar[returned_points_indices[j]]))
            for l in range(F):
                probs[i,digit] = probs[i,digit] + ((((np.power(-1, l)) / (z - z_bar[returned_points_indices[l]]))/den)*Y[returned_points_indices[l],digit])

    return probs





def model_out(Y):
    n=len(Y)
    outputs=np.zeros([n,10])
    #trans = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])
    for i in range(n):
        data=Y[i,]
        torch_sample = torch.from_numpy(data).float()
        #torch_sample = torch_sample.unsqueeze(0)
        #torch_sample = torch_sample.permute( 2, 0, 1)
        #torch_sample = trans(torch_sample)
        torch_sample = torch_sample.unsqueeze(0)
        torch_sample = torch_sample.unsqueeze(0)
        outputs[i,]= model(torch_sample).detach().numpy()[0]

        # _, predicts = torch.max(pred, 1)
        # predictions[i]=predicts.numpy()[0]

    return outputs


def Determine_accuracy(input_batch_ids):
    input_batch=test_x[input_batch_ids]
    n=len(input_batch)
    predictions=np.zeros([n,1])
    #trans = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])
    for i in range(n):
        data=input_batch[i,]
        torch_sample = torch.from_numpy(data).float()
        #torch_sample = torch_sample.unsqueeze(0)
        #torch_sample = torch_sample.permute( 2, 0, 1)
        #torch_sample = trans(torch_sample)
        torch_sample = torch_sample.unsqueeze(0)
        torch_sample = torch_sample.unsqueeze(0)
        pred = model(torch_sample)
        _, predicts = torch.max(pred, 1)
        predictions[i]=predicts.numpy()[0]

    test_y_samples=test_y[input_batch_ids]
    test_y_samples_reshaped=np.reshape(test_y_samples, [n, 1])
    diff=test_y_samples_reshaped-predictions
    # print("Accuracy is")
    out=(n-np.count_nonzero(diff))

    return out

##### Unifying parameters
class parameter_container:
    def __init__(self):
        self.K= 1
        self.N =1
        self.S =1
        self.emax=1
        #self.actual_num_of_errors=1
        self.alpha = np.zeros(self.K) # function interpolation points
        self.z_bar = np.zeros(self.N) # function evaluation points
        self.beta=np.zeros(self.emax)



    def update_parameters(self, K, N, S):
        self.K= K
        self.N =N
        #self.emax=int((N-K-1)/2)
        self.emax = 2
        self.S =S
        #self.actual_num_of_errors=actual_num_0f_errors

        self.alpha = np.zeros(self.K)
        for j in range(self.K):
            self.alpha[j] = np.cos(((2 * j + 1) * np.pi) / (2 * self.K))

        self.z_bar = np.zeros(N)
        for i in range(self.N):
            self.z_bar[i] = np.cos((i * np.pi) / (self.N))  #####


        # self.beta= np.zeros(self.emax+1)
        # for i in range(self.emax+1):
        #     self.beta[i] = np.cos((i * np.pi+1) / (self.emax+1))  #####

        self.beta= np.zeros(self.emax+1)
        for i in range(self.emax+1):
            self.beta[i] = np.cos(((2* i + 1) * np.pi) / (2 * (self.emax+1))) #####

        # self.beta = np.zeros(self.emax+1) ### t=emax+1
        # self.beta=[-.83,.03,.83]


def plot_parameters():
    plt.plot(parameters.alpha,np.zeros((parameters.K)), marker='x', markersize=8, linewidth=0)
    plt.plot(parameters.z_bar, np.zeros((parameters.N)),  marker='+', markersize=8, linewidth=0)
    plt.plot(parameters.beta, np.zeros((parameters.emax+1)),  marker='1', markersize=8, linewidth=0)
    plt.show()


def Acc_Comparison(K, N, S, iterations,actual_num_0f_errors):
    parameters.update_parameters(K,N,S)### sets parameters

    #plot_parameters()
    Berrut_Accuracy = 0
    Centralized_accuracy = 0

    for i in range(iterations):
        # Random data
        shuffled_indices=np.random.permutation(test_x.shape[0])
        random_indices = shuffled_indices[0:K]
        random_indices=np.sort(random_indices)
        test_sample_x = test_x[random_indices]
        True_labels = test_y[random_indices]


        #dataset sweep


        # Centralized Accuracy
        # centralized_test_sample_x = tf.expand_dims(test_sample_x, 3)
        # probs_centralized = new_model.predict(centralized_test_sample_x)
        # centralized_predictions = np.argmax(probs_centralized, axis=1)
        Single_Centralized_Accuracy=Determine_accuracy(random_indices)
        Centralized_accuracy = Centralized_accuracy +Single_Centralized_Accuracy / K
        # Distributed Inference
        # encoding test data
        coded_test_sample_x = encoder(test_sample_x)

        # train_x=tf.expand_dims(train_x,3)
        # test_x=tf.expand_dims(test_x,3)

        model_outputs=model_out(coded_test_sample_x)

        ## Determining stragglers' indices ####
        returned_points_indices = np.random.permutation(N)
        returned_points_indices = returned_points_indices[0:N - S]
        returned_points_indices = np.sort(returned_points_indices)
        # returned_points_indices=range(N-S)
        # returned_points_indices=range(N)



        test_sample_out_value = decoder(model_outputs, returned_points_indices,actual_num_0f_errors)
        #print(test_sample_out_value.shape)
        Berrut_predictions = np.argmax(test_sample_out_value, axis=1)

        # Perfomance Evaluation
        #True_labels = test_sample_y
        Berrut_predictions=Berrut_predictions.reshape(K, 1)
        True_labels_reshaped=np.reshape(True_labels,[len(True_labels),1])
        Berrut_Accuracy = Berrut_Accuracy + np.count_nonzero(Berrut_predictions - True_labels_reshaped) / K
        print("%" + str((i + 1) * 100 / iterations) + "completed")


    return 1 - Berrut_Accuracy / iterations,  Centralized_accuracy / iterations


def Plot_N():
    K=10
    S=1
    #N=np.arange(43,81,2)
    N=[11,13,14,15,17]
    #N=[5,7,9]
    Berrut_acc=np.zeros(len(N))
    Center_acc=np.zeros(len(N))
    num_of_iterations=100


    for i in range(len(N)):
        print("N="+str(N[i]))
        a,b=Acc_Comparison(K,N[i],S,num_of_iterations,0)
        Berrut_acc[i]=a
        Center_acc[i]=b

    np.savetxt("Berrut_acc.txt", Berrut_acc)
    np.savetxt("Center_acc.txt", Center_acc)
    np.savetxt("N.txt", N)
    plt.plot(N, Berrut_acc, label="Berrut")
    plt.plot(N, Center_acc, label="Centralized")
    plt.legend(["Berrut", "Centralized"])
    plt.xlabel("N")
    plt.ylabel("Accuracy")
    plt.title('K=,'+str(K)+' S=' + str(S) + ', Num_of_Iterations=' + str(num_of_iterations))
    plt.show()



# Plot vs K
def Plot_K():
    #K=np.arange(2,14,2)
    #S=3

    #K=np.arange(14,24,2)
    #K = np.arange(7, 11, 2)
    #K=np.arange(4,8,2)
    K=np.arange(8, 14, 2)
    e_max=2
    actual_num_0f_errors =2

    #c_err=0 ### number of components in error locator function
    S=0
    Berrut_acc=np.zeros(len(K))
    Center_acc=np.zeros(len(K))
    num_of_iterations=100


    for i in range(len(K)):
        print("K="+str(K[i]))
        a,b=Acc_Comparison(K[i],2*K[i]+2*e_max+S+1,S,num_of_iterations,actual_num_0f_errors)
        Berrut_acc[i]=a
        Center_acc[i]=b

    train_your_nicest_model(1)

    np.savetxt("Berrut_acc.txt",Berrut_acc)
    np.savetxt("Center_acc.txt",Center_acc)
    np.savetxt("K.txt",K)
    plt.plot(K,Berrut_acc,label="Berrut")
    plt.plot(K,Center_acc,label="Centralized")
    plt.legend(["Berrut","Centralized"])
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.title('N=K+S+2*e_max, S=' +str(S)+',e_max=' +str(e_max)+', actual=' +str(actual_num_0f_errors)+',Num_of_Iterations='+ str(num_of_iterations))
    plt.show()


###############  START HERE ##############


#initiate model parameter object
parameters = parameter_container()
#model_accuracy_pytorch()
#model_accuracy()
Plot_K()
#train_your_nicest_model(1)