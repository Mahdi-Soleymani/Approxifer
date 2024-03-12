import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib as tkz
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl

#plt.style.use("ggplot")
########### Comparison   Strgglers ONLY ############################################
def parM_comparison():
    N = 3
    K=12
    #fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

    centralized = (.99, .97, .94)
    Berrut8=(.872, .74, .776)#k=8
    ParM8= (.6833, .6711, .2539)#k=8
    Berrut10=(.872, .73, .774)#k=10
    ParM10 = (.592, .59, .2043)#k=10
    Berrut12=(.868, .72, .774)#k=12
    ParM12 = (.5031, .5518, .1921)#k=12

    width = 0.25
    ind = np.arange(N)
    names=('MNIST', 'Fashion-MNIST', 'CIFAR10')


    plt.bar(ind, centralized, width, label='Base model (Best case)',zorder=3)
    plt.bar(ind + width, Berrut8, width,
        label='ApproxIFER',zorder=3)
    plt.bar(ind + 2*width, ParM8, width,
        label='ParM',zorder=3)

    plt.ylabel('Accuracy', fontsize=18)
    #plt.title('K='+str(K),x=.95, y=1.01)

    plt.xticks(ind + width / 2, ('MNIST', 'Fashion-MNIST', 'CIFAR10'),fontsize=18)
    #plt.legend(bbox_to_anchor=(0, 1.1), loc='upper left', ncol=3)
    plt.legend(bbox_to_anchor=(1.07, 1.13), ncol=3, fontsize=12)

    plt.grid(True,axis='y',zorder=0, color='grey')
    plt.show()
    #plt.savefig("ParM8.pdf")




######## Stragglers S=1,2,3
def extra_straggler():
    N =3
    K=8
    Centralized=(.99, .97, .94)
    BerrutS1 = (.868, .72, .774)
    BerrutS2 = (.8256, .6554, .7405)
    BerrutS3 = (.7747, .64, .7305)



    ind = np.arange(N)
    width = 0.15
    plt.bar(ind, Centralized, width, label='Base model (Best case)',zorder=3)
    plt.bar(ind+width, BerrutS1, width, label='S=1',zorder=3)
    plt.bar(ind + 2*width, BerrutS2, width,
    label='S=2',zorder=3)
    plt.bar(ind + 3*width, BerrutS3, width,
    label='S=3',zorder=3)

    plt.ylabel('Accuracy',fontsize=18)
    #plt.title('K='+str(K),x=.95, y=1.02)

    plt.xticks(ind + width / 2, ('MNIST', 'Fashion-MNIST', 'CIFAR10'),fontsize=18)
    #plt.legend(bbox_to_anchor=(0, 1.1), loc='upper left', ncol=4)
    plt.legend(bbox_to_anchor=(1.11, 1.13), ncol=4, fontsize=12)
    plt.grid(True,axis='y',zorder=0, color='grey')
    plt.show()
# plt.savefig('destination_path.eps', format='eps')

############# Strglers COMPLEX models Comparison #############
#
def complex_straggler():
    N = 5
    K=10
    centralized = (.94,.9334,.9365,.9407, .9285 )
    # Berrut=()#k=8
    # Berrut=()#k=12
    Berrut=(.805, .812, .801,.817,.77 )#k=10



    ind = np.arange(N)
    width = 0.25
    plt.bar(ind, centralized, width, label='Base model (Best case)',zorder=3)
    plt.bar(ind + width, Berrut, width,
        label='ApproxIFER ',zorder=3)
    #plt.bar(ind + 2*width, ParM, width,label='ParM')

    plt.ylabel('Accuracy',fontsize=18)
    #plt.title('K='+str(K),x=.95, y=1.01)

    plt.xticks(ind + width / 2, ('VGG16','ResNet34', 'ResNet50', 'DenseNet161',  'GoogLeNet'),fontsize=11.5)
    plt.legend(bbox_to_anchor=(-.035, 1.13), loc='upper left', ncol=3,fontsize=14)
    plt.grid(True, axis='y', zorder=0, color='grey')
    plt.show()



######## ERRORS e=1,2,3

def error_123():
    N =3
    K=12
    Centralized=(.99, .93, .94)
    BerrutS1 = (.97, .79, .8981)
    BerrutS2 = (.95, .805, .8966)
    BerrutS3 = (.93, .82, .901)



    ind = np.arange(N)
    width = 0.15
    plt.bar(ind, Centralized, width, label='Base model (Best case)',zorder=3)
    plt.bar(ind+width, BerrutS1, width, label='E=1',zorder=3)
    plt.bar(ind + 2*width, BerrutS2, width,
        label='E=2',zorder=3)
    plt.bar(ind + 3*width, BerrutS3, width,
        label='E=3',zorder=3)

    plt.ylabel('Accuracy',fontsize=18)
    #plt.title('K='+str(K),x=.95, y=1.02)

    plt.xticks(ind + 3*width / 2, ('MNIST', 'Fashion-MNIST', 'CIFAR10'),fontsize=18)
    plt.legend(bbox_to_anchor=(-.07, 1.12), loc='upper left', ncol=4,fontsize=11)
    plt.grid(True, axis='y', zorder=0, color='grey')

    plt.show()




############ Error COMPLEX models Comparison #############
#
def error_complex():
    beingsaved = plt.figure()
    N = 5
    K=12
    centralized = (.94,.9334,.9365,.9407, .9285 )
    # Berrut=()#k=8
    # Berrut=()#k=12
    Berrut=(.9, .8933, .8993,.9070,.8824 )#k=10



    ind = np.arange(N)
    width = 0.25
    plt.bar(ind, centralized, width, label='Base model (Best case)',zorder=3)
    plt.bar(ind + width, Berrut, width,
        label='ApproxIFER ',zorder=3)
    #plt.bar(ind + 2*width, ParM, width,label='ParM')

    plt.ylabel('Accuracy', fontsize=18)
    #plt.title('K='+str(K),x=.95, y=1.01)

    plt.xticks(ind + width / 2, ('VGG16','ResNet34', 'ResNet50', 'DenseNet161',  'GoogLeNet'),fontsize=11.5)
    plt.legend(bbox_to_anchor=(-0.035, 1.25), loc='upper left', ncol=1, fontsize=18)
    plt.grid(True, axis='y', zorder=0, color='grey')

    plt.show()
    beingsaved.savefig('destination_path.eps', format='eps', dpi=1000)
    #tkz.save("test.tex")
    #plt.savefig("fig1.eps", format='eps')



def error_sig():
    N = 2
    #K=12
    #fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

    sig1 = (.9607, .8012)
    sig10=(.9434, .7723)#k=8
    sig100= (.9690, .8037)#k=8

    Berrut10=(.872, .73, .774)#k=10
    ParM10 = (.592, .59, .2043)#k=10
    Berrut12=(.868, .72, .774)#k=12
    ParM12 = (.5031, .5518, .1921)#k=12

    width = 0.25
    ind = np.arange(N)
    names=('MNIST', 'Fashion-MNIST')


    plt.bar(ind, sig1, width, label='$\sigma$=1',zorder=3)
    plt.bar(ind + width, sig10, width,
        label='$\sigma$=10',zorder=3)
    plt.bar(ind + 2*width, sig100, width,
        label='$\sigma$=100',zorder=3)

    plt.ylabel('Accuracy', fontsize=18)
    #plt.title('K='+str(K),x=.95, y=1.01)

    plt.xticks(ind + width / 2, ('MNIST', 'Fashion-MNIST'),fontsize=18)
    #plt.legend(bbox_to_anchor=(0, 1.1), loc='upper left', ncol=3)
    plt.legend(bbox_to_anchor=(1.00, 1.16), ncol=3, fontsize=16)

    plt.grid(True,axis='y',zorder=0, color='grey')
    plt.show()
    #plt.savefig("ParM8.pdf")


########################## MAIN
#parM_comparison()
#extra_straggler()
#complex_straggler()
#error_123()
#error_complex()
error_sig()