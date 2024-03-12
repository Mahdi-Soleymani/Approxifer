import torch
import torchvision
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
import torch
from torchvision import datasets, transforms as T
import torchvision.models as models
from logger import logger
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from nppretty import ArrayStream
import time
import os
from knockknock import telegram_sender
x=1
CHAT_ID: int = 43515446
@telegram_sender(token="1834099231:AAEHY1G5pAGDRXH20vyNuP-WrfUqbc4f-X8", chat_id=CHAT_ID)
def train_your_nicest_model(your_nicest_parameters):
    return {'loss': 0.9} # Optional return value

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])



# dataset = CIFAR10(root='data/', download=True, transform=ToTensor())
# test_dataset = CIFAR10(root='data/', train=False, transform=ToTensor())
#dataset = CIFAR10(root='data/', download=True, transform=transform)

#test_dataset = CIFAR10(root='data/', train=False, transform=transform)
#validation_dir='validation/validation/ILSVRC2012_img_val/'


# validation_dir='../../validation/validation'
# validation_dataset=torchvision.datasets.ImageFolder(root=validation_dir,transform = transform)

batch_size=128

# train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
# val_loader = DataLoader(val_ds, batch_size, num_workers=0, pin_memory=True)


# validation_loader = DataLoader(validation_dataset, batch_size, num_workers=0, pin_memory=True)
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


############ model and data selection   ##########
#model = models.resnet18(pretrained=True)
model=models.resnet50(pretrained=True)
model.eval()
num_of_classes=1000
top_what_accuracy=5
def model_accuracy_pytorch():
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in validation_loader:
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
        print(i)

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

    probs=np.zeros([K,num_of_classes])
    for digit in range(num_of_classes):
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
    outputs=np.zeros([n,num_of_classes])
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
    input_batch=test_x[input_batch_ids,:,:,:]
    n=len(input_batch)
    predictions=np.zeros([n,top_what_accuracy])
    #trans = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])
    if top_what_accuracy==1:
        for i in range(n):
            data=input_batch[i,]

            torch_sample = torch.from_numpy(data).float()
            torch_sample = torch_sample.permute(2, 0, 1)
            torch_sample = torch_sample.unsqueeze(0)
            pred = model(torch_sample)
            _, predicts = torch.max(pred, 1)
            predictions[i]=predicts.numpy()[0]

        diff=test_y[input_batch_ids]-predictions
        # print("Accuracy is")
        out=(n-np.count_nonzero(diff))

    else: ## top_what_accuracy=5
        for i in range(n):
            data = input_batch[i,]
            torch_sample = torch.from_numpy(data).float()
            torch_sample = torch_sample.permute(2, 0, 1)
            torch_sample = torch_sample.unsqueeze(0)
            pred = model(torch_sample)
            _, predicts = torch.topk(pred, top_what_accuracy)
            # _, predicts = torch.max(pred, 1)
            predictions[i,] = predicts.numpy()[0]
        out=0
        for i in range(len(input_batch_ids)):
            if test_y[input_batch_ids[i]] in predictions[i,]:
                out=out+1

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
        Centralized_accuracy = Centralized_accuracy +(Single_Centralized_Accuracy / K)

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
        #Berrut_predictions = np.argmax(test_sample_out_value, axis=1)
        Berrut_predictions = np.argsort(test_sample_out_value, axis=1)
        Berrut_predictions_top=Berrut_predictions[:,-top_what_accuracy:]
        Berrut_predictions_top = Berrut_predictions_top.reshape(K, top_what_accuracy)
        # Perfomance Evaluation
        #True_labels = test_sample_y
        temp=0
        for j in range(K):
            if True_labels[j] in Berrut_predictions_top[j,]:
                temp=temp+1
        #Berrut_predictions=Berrut_predictions.reshape(K, 1)
        Berrut_Accuracy=Berrut_Accuracy+temp/K
        print("%" + str((i + 1) * 100 / iterations) + "completed")


    return Berrut_Accuracy / iterations,  Centralized_accuracy / iterations


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
    #K=np.arange(2,15,1)
    K=[2,4,6,8,10,12]
    #K=[21, 27, 35]
    S=1

    Berrut_acc=np.zeros(len(K))
    Center_acc=np.zeros(len(K))
    num_of_iterations=1000



    for i in range(len(K)):
        print("K="+str(K[i]))
        a,b=Acc_Comparison(K[i],K[i]+S,S,num_of_iterations)
        Berrut_acc[i]=a
        Center_acc[i]=b

    model_name = model.__class__.__name__+'50'
    os.chdir('ImageNet_Results')
    os.makedirs(model_name, exist_ok=True)
    train_your_nicest_model(1)
    os.chdir(model_name)
    np.savetxt('K='+str(K)+'S='+str(S)+model_name+'iterations'+str(num_of_iterations)+"top"+str(top_what_accuracy)+"Berrut_acc.txt",Berrut_acc)
    np.savetxt('K='+str(K)+'S='+str(S)+model_name+'iterations'+str(num_of_iterations)+"top"+str(top_what_accuracy)+"Center_acc.txt",Center_acc)
    np.savetxt('K='+str(K)+'S='+str(S)+model_name+'iterations'+str(num_of_iterations)+"top"+str(top_what_accuracy)+"K.txt",K)
    plt.plot(K,Berrut_acc,label="Berrut")
    plt.plot(K,Center_acc,label="Centralized")
    plt.legend(["Berrut","Centralized"])
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.title('N=K+1, S=' +str(S)+', Num_of_Iterations='+ str(num_of_iterations))
    plt.show()



#### Import data from Keras
#(train_x,train_y), (test_x,test_y)=K.datasets.cifar10.load_data()
# (train_x,train_y), (test_x,test_y)=K.datasets.cifar10.load_data()
# train_x=train_x/255.0
# test_x=test_x/255.0

###Loading labels

x = []
file_in = open('ImageNet_validation_labels_consistent_with_pytorch.txt', 'r')
for y in file_in.read().split('\n'):
    x.append(int(y))
test_y= np.array(x)
print('Labels are loaded')
test_y=test_y.reshape([50000,1])
print(test_y.shape)

# a=int(50000.0/batch_size)
# a=(a+1)*batch_size
test_x=np.zeros([50000,224,224,3])


print('loading dataset . . . ')
t_start=time.time()
test_x=np.load('validation_dataset_in_numpy_format.npy')
t_end=time.time()
print('It took ' + str(t_end-t_start)+' to load dataset')
test_x=test_x[:50000]
print(test_x.shape)

########### change data to numpy
#id=0
# for item in validation_loader:
#     images,_=item
#     images=images.permute(0,2,3,1)
#     images_numpy=images.numpy()
#     test_x[id:id+batch_size,:,:,:]=images_numpy
#     id=id+batch_size
#     print(id)
#     # print(images[1,:,:,0])
#     # break
# np.save('validation_dataset_in_numpy_format',test_x)
# print('file is saved')

# print(test_x[1,:,:,0])
# plt.imshow(images[1,:,:,:])
# plt.show()
#model_accuracy_pytorch()
#model_accuracy()
Plot_K()
