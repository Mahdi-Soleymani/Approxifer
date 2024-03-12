import torch
#import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR100
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



transform = transforms.Compose([
        #transforms.Resize(256),
        #transforms.RandomCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        #transforms.Normalize((0.5), (0.25))
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        ])

# dataset = CIFAR10(root='data/', download=True, transform=ToTensor())
# test_dataset = CIFAR10(root='data/', train=False, transform=ToTensor())
#dataset = CIFAR10(root='data/', download=True, transform=transform)

test_dataset = CIFAR10(root='data/', train=False, transform=transform)


batch_size=128

# train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
# val_loader = DataLoader(val_ds, batch_size, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size, num_workers=0, pin_memory=True)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


############ model and data selection   ##########
model=resnet.ResNet18()
PATH='../base_model_trained_files/cifar10/resnet18/model.t7'
model.load_state_dict(torch.load(PATH))
model.eval()


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
         100*correct/total ))
    print(correct, total)


def model_accuracy():
    n = len(test_x)
    predictions = np.zeros([n, 1])
    # trans = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])
    for i in range(n):
        data = test_x[i,]
        torch_sample = torch.from_numpy(data).float()
        torch_sample = torch_sample.permute(2, 0, 1)
        # torch_sample = trans(torch_sample)
        torch_sample = torch_sample.unsqueeze(0)
        pred = model(torch_sample)
        _, predicts = torch.max(pred, 1)
        predictions[i] = predicts.numpy()[0]

    #diff = test_y - predictions
    diff=test_y-predictions
    print("Model accuracy is")
    out = (n - np.count_nonzero(diff))/n
    print(out)
    print(n-np.count_nonzero(diff))

    return out


#Berrut Encoder
def encoder(X,N):
    [K,H,W,C]=np.shape(X)
    alpha=np.zeros(K)
    for j in range(K):
        alpha[j]=np.cos(((2*j+1)*np.pi)/(2*K))

    all_z=np.zeros(N)
    for i in range(N):
        all_z[i]=np.cos((i*np.pi)/N)

    coded_X=np.zeros([N,H,W,C])
    for n in range(N):
        z=all_z[n]
        den=0
        for j in range(K):
            den = den+(np.power(-1, j)) / (z - alpha[j])
        for i in range(K):
            coded_X[n,]=coded_X[n,]+(((np.power(-1, i)) / (z - alpha[i]))/den)*X[i,]
    return coded_X




#Berrut Decoder
def decoder(Y,K,N,returned_points_indices):
    F=len(returned_points_indices)
    alpha=np.zeros(K)
    for j in range(K):
        alpha[j]=np.cos(((2*j+1)*np.pi)/(2*K))

    z_bar=np.zeros(N)
    for i in range(N):
        z_bar[i]=np.cos((i*np.pi)/N)

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





def model_out(Y):
    n=len(Y)
    outputs=np.zeros([n,10])
    #trans = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])
    for i in range(n):
        data=Y[i,]
        torch_sample = torch.from_numpy(data).float()
        #torch_sample = torch_sample.unsqueeze(0)
        torch_sample = torch_sample.permute( 2, 0, 1)
        #torch_sample = trans(torch_sample)
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
        torch_sample = torch_sample.permute( 2, 0, 1)
        torch_sample = torch_sample.unsqueeze(0)
        pred = model(torch_sample)
        _, predicts = torch.max(pred, 1)
        predictions[i]=predicts.numpy()[0]

    diff=test_y[input_batch_ids]-predictions
    # print("Accuracy is")
    out=(n-np.count_nonzero(diff))
    return out



def Acc_Comparison(K, N, S, iterations):
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
        coded_test_sample_x = encoder(test_sample_x, N)

        # train_x=tf.expand_dims(train_x,3)
        # test_x=tf.expand_dims(test_x,3)

        model_outputs=model_out(coded_test_sample_x)

        ## Determining stragglers' indices ####
        returned_points_indices = np.random.permutation(N)
        returned_points_indices = returned_points_indices[0:N - S]
        returned_points_indices = np.sort(returned_points_indices)
        # returned_points_indices=range(N-S)
        # returned_points_indices=range(N)
        test_sample_out_value = decoder(model_outputs, K, N, returned_points_indices)
        #print(test_sample_out_value.shape)
        Berrut_predictions = np.argmax(test_sample_out_value, axis=1)

        # Perfomance Evaluation
        #True_labels = test_sample_y
        Berrut_predictions=Berrut_predictions.reshape(K, 1)
        Berrut_Accuracy = Berrut_Accuracy + np.count_nonzero(Berrut_predictions - True_labels) / K
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
    num_of_iterations=1


    for i in range(len(N)):
        print("N="+str(N[i]))
        a,b=Acc_Comparison(K,N[i],S,num_of_iterations)
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
    #K=np.arange(3,15,2)
    K = np.arange(14, 24, 2)
    S=1

    Berrut_acc=np.zeros(len(K))
    Center_acc=np.zeros(len(K))
    num_of_iterations=2000


    for i in range(len(K)):
        print("K="+str(K[i]))
        a,b=Acc_Comparison(K[i],K[i]+S,S,num_of_iterations)
        Berrut_acc[i]=a
        Center_acc[i]=b

    train_your_nicest_model(1)

    np.savetxt("Berrut_acc.txt",Berrut_acc)
    np.savetxt("Center_acc.txt",Center_acc)
    np.savetxt("N.txt",K)
    plt.plot(K,Berrut_acc,label="Berrut")
    plt.plot(K,Center_acc,label="Centralized")
    plt.legend(["Berrut","Centralized"])
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.title('N=K+1, S=' +str(S)+', Num_of_Iterations='+ str(num_of_iterations))
    plt.show()

#### Import data from Keras
#(train_x,train_y), (test_x,test_y)=K.datasets.cifar10.load_data()
(train_x,train_y), (test_x,test_y)=K.datasets.cifar10.load_data()
train_x=train_x/255.0
test_x=test_x/255.0


#manual normalization
# mean=[0.5, 0.5, 0.5]
# std=[0.25, 0.25, 0.25]
mean=[0.4914, 0.4822, 0.4465]
std=[0.2471, 0.2435, 0.2616]
test_x=(test_x-mean)/std




#model_accuracy_pytorch()
#model_accuracy()
Plot_K()


