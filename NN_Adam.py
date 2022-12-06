import pickle
import pandas as pd
from json.encoder import INFINITY
import numpy as np
import random
import math
import matplotlib.pyplot as plt 
import seaborn as sns
import tracemalloc
import time

 


def error_rate(Y,pred):
    err = 0
    for i in range(len(Y)):
        if Y[i] != pred[i]:
            err+=1
    return err/len(Y)


def accuracy_score(Y,pred):
    err = 0
    for i in range(len(Y)):
        if Y[i] != pred[i]:
            err+=1
    return 1 - (err/len(Y))

def confusion_matrix(Y, pred):
    matrix=np.zeros((2,2)) 
    for i in range(len(pred)):

        if int(pred[i])==1 and int(Y[i])==1: 
            matrix[1,1]+=1 
        elif int(pred[i])==1 and int(Y[i])==0: 
            matrix[0,1]+=1 
        elif int(pred[i])==0 and int(Y[i])==1:
            matrix[1,0]+=1 
        elif int(pred[i])==0 and int(Y[i])==0:
            matrix[0,0]+=1 

    return matrix

def classification_report(y, pred):

    confMat = confusion_matrix(y, pred)
    data = np.zeros_like(confMat).astype(float)

    data[0][0] = round(confMat[0][0]/(confMat[0][0] + confMat[1][0]),2)

    data[1][0] = round(confMat[1][1]/(confMat[1][1] + confMat[0][1]),2)

    data[0][1] = 1 - data[0][0] 

    data[1][1] = 1 - data[1][0]

    df = pd.DataFrame(data, columns=['Precision', 'Error rate'])
    print(df)
    return df





def h(x):
    return math.sin(2*math.pi*x)
def hlin(a):
    return a
def hplin(a):
    return 1
def hsig(a):
    return -1+2 / (1 + np.exp(-a))
def hpsig(a):
    return 2*np.exp(-a) / ((1 + np.exp(-a))**2)

    



class scaler():
    def __init__(self,X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
    def scaleData(self,X):
        return (X-self.mean)/self.std
    def normalizeData(self,X):
        return (X-self.min)/(self.max-self.min)


class adamNN:
    def __init__(self,hNodes,batchsz = 32,alpha= 1/1000):
        self.weights1 = []
        self.weights2 = []
        self.hNodes = hNodes
        self.batchSize = batchsz
        self.trainerror = []
        self.valerror = []
        self.learnR = alpha
        self.trainAccuracy = []
        self.valAccuracy = []

    def makeBasis(self,input):
        ones = np.array([[1]]*len(input))
        arr = np.concatenate([ones, input], axis=1)
        return arr

    def test(self,input,target):
        totRight = 0
        totalSize = len(target)
        featuresLen = len(input[0,:])
        self.hNodes = len(self.weights1)
        error = 0
        for n in range(totalSize):
            x = input[n,:]
            t = target[n]
            # forward propagate
            y = np.zeros(self.hNodes+1)
            y[0] = 1
            for j in range(self.hNodes):
                a = 0
                for k in range(featuresLen):
                    a += self.weights1[j,k]*x[k]
        
                y[j+1] = hsig(a)
            a = 0
            for j in range(self.hNodes+1):
                a += self.weights2[0,j]*y[j]
            z = hlin(a)
            error += (z-t)**2
            z2 = int(hlin(a).round())
            if z2 == t:
                totRight +=1
            

        return error/len(target), totRight/len(target)




    def predictBased(self,x):
        x = self.makeBasis(x)
        self.hNodes = len(self.weights1)
        
        featuresLen = len(self.weights1[0])
        
    
        L = len(x);
        z = np.zeros(L);

        for i in range(L):

            # forward propagate
            inpNode = x[i]
            y = np.zeros(self.hNodes+1)

            y[0] = 1
            for j in range(self.hNodes):
                a = 0
                for k in range(featuresLen):
                    vv = self.weights1[j,k]*inpNode[k]
                    a =  a+ vv
                val = hsig(a)
                y[j+1] = val

            
            a = 0
            for j in range(self.hNodes+1):
                a += self.weights2[0,j]*y[j]
    
            z[i] = hlin(a)
        return z



    def predictBasedInt(self,x):
        x = self.makeBasis(x)
        self.hNodes = len(self.weights1)
        
        featuresLen = len(self.weights1[0])
        
    
        L = len(x);
        z = np.zeros(L);

        for i in range(L):

            # forward propagate
            inpNode = x[i]
            y = np.zeros(self.hNodes+1)

            y[0] = 1
            for j in range(self.hNodes):
                a = 0
                for k in range(featuresLen):
                    vv = self.weights1[j,k]*inpNode[k]
                    a =  a+ vv
                val = hsig(a)
                y[j+1] = val

            
            a = 0
            for j in range(self.hNodes+1):
                a += self.weights2[0,j]*y[j]

            z[i] = int(hlin(a).round())

        return z

    def train(self,input, target, maxEpochs = INFINITY):
        input = np.array(input)
        target = np.array(target)
        




        
        # decay rate
        beta1 = 0.9
        beta2 = 0.999

        # prevent zero division
        epsilon = 1e-8





        input  = self.makeBasis(input)
        

        # total samples
        totalSize = len(target)
        # feature size
        featuresLen = len(input[0,:])



        # validation number
        holdNN = int(round( totalSize/3))

        # total to train
        totTrain = totalSize - holdNN


        # randomly shuffle the inputs and targets
        rnn = list(range(totalSize))
        random.shuffle(rnn)
        idx = rnn

        # split 1/3 for validation and 2/3 for training
        train_input = input[idx[0:totTrain],:]
        train_target = target[idx[0:totTrain]]

        val_input = input[idx[(totTrain+1):totalSize],:]
        val_target = target[idx[(totTrain+1):totalSize]]

        # Momentum parameters 
        m1 = np.zeros((self.hNodes, featuresLen))
        m2 = np.zeros((1,self.hNodes+1))
        v1 = np.zeros((self.hNodes, featuresLen))
        v2 = np.zeros((1,self.hNodes+1))



        # input activation
        inpNode = np.zeros(featuresLen)
        # hidden activation
        hdnNode = np.zeros(self.hNodes)
        # output activation
        outNode = 0




        # initializing random layer 1 weights
        self.weights1 = 5*np.random.randn(self.hNodes, featuresLen)  
        # initializing random layer 2 weights (inc bias)
        self.weights2 = 5*np.random.randn(1,self.hNodes+1)

        numweights = (len(self.weights1)*len(self.weights1[0])) + (len(self.weights2)*len(self.weights2[0]))
        print(str(numweights)+" weights")
        
        init_train_error,init_train_acc = self.test( train_input, train_target)
        print("Initial Training Error = "+str(init_train_error))

        init_val_error,init_val_acc = self.test( val_input, val_target)
        print("Initial Validation Error = "+str(init_val_error))


        epoch = 1
        updateVal = INFINITY


        # stop training if converges or max epochs are reached
        while epoch <= maxEpochs and updateVal > 1E-6:

            oldWeights1 = np.copy(self.weights1)
            oldWeights2 = np.copy(self.weights2)
            
            iter = 1
            offset = -1
            grad1 = np.zeros((self.hNodes, featuresLen))
            grad2 = np.zeros((1,self.hNodes+1))

            # shuffle data for this epoch
            rnn = list(range(totTrain))
            random.shuffle(rnn)
            trainidx = rnn
            # looping over mini-batches
            for xx  in range(int(math.floor(totTrain/self.batchSize)-1)):
                offset += self.batchSize
                for n in range(self.batchSize):

                    # input and target
                    x = train_input[trainidx[offset + n],:]
                    t = train_target[trainidx[offset + n]]
                    
                    # starting forward propagate
                    inpNode = x
                    y = np.zeros(self.hNodes+1)

                    y[0] = 1
                    for j in range(self.hNodes):
                        hdnNode[j] = 0
                        for k in range(featuresLen):
                            hdnNode[j] += self.weights1[j,k]*inpNode[k]
                        
                        y[j+1] = hsig(hdnNode[j])
                    

                    outNode = 0
                    for j in range(self.hNodes+1):
                        outNode += self.weights2[0,j]*y[j]
                    z = hlin(outNode)
                    
                  
                    
                    # calculating delta (output error)
                    delta = (z-t)

                    # calculating layer 2 gradients by backpropagation of delta
                    for j in range(self.hNodes+1):
                        if j == 0:
                            grad2[0,j] += delta*y[j]
                        else:
                            grad2[0,j] += delta*y[j]*hplin(outNode)
                                
                    

                    # calculating layer 1 gradients by backpropagation of delta
                    for i in range(featuresLen):
                        for j in range(self.hNodes):
                            grad1[j,i] += delta*self.weights2[0,j+1]*hpsig(hdnNode[j])*x[i]
                        
                    
                

                # updating moment estimates
                m1 = beta1 * m1 + (1-beta1) *grad1
                m2 = beta1 * m2 + (1-beta1) *grad2
                
                v1 = beta2*v1 + (1-beta2)*(grad1**2)
                v2 = beta2*v2 + (1-beta2)*(grad2**2)
                
                m1hat = m1/(1-beta1**iter)
                m2hat = m2/(1-beta1**iter)
                
                v1hat = v1/(1-(beta2**iter))
                v2hat = v2/(1-(beta2**iter))

                iter += 1
                
                # update layer 2 weights with adam parameters
                for j in range(self.hNodes+1):
                    update = self.learnR*m2hat[0,j]/(math.sqrt(v2hat[0,j]) + epsilon)
                    self.weights2[0,j] -= update
                
                
                # update layer 1 weights with adam parameters
                for i in range(featuresLen):
                    for j in range(self.hNodes):
                        update = self.learnR*m1hat[j,i]/(math.sqrt(v1hat[j,i]) + epsilon)
                        self.weights1[j,i] -= update
                    
                
            

            temper, acc = self.test( train_input, train_target)

            self.trainerror.append(temper)
            self.trainAccuracy.append(acc)
            
            temper1, acc1 = self.test(val_input, val_target)
        
            self.valerror.append(temper1)
            self.valAccuracy.append(acc1)      
                      
            # calculating update magnitude
            updateVal = 0
            for j in range(self.hNodes+1):
                dw = oldWeights2[0,j] - self.weights2[0,j]
                updateVal += dw*dw
            
            for i in range(featuresLen):
                for j in range(self.hNodes):
                    dw = oldWeights1[j,i] - self.weights1[j,i]
                    updateVal += dw*dw
                
            
            
            print("Epoch "+str(epoch)+"/"+str(maxEpochs)+"----------error: "+str(round(temper, 4))+ " - accuracy: "+str(round(acc, 4))+ " - val_error: "+str(round(temper1, 4))+ " - val_accuracy: "+str(round(acc1, 4)))

            # increasing epoch
            epoch += 1
        
        
        return self.weights1, self.weights2, self.test(val_input, val_target)






def load_data(X,y, train_size=0.9):
    X = np.concatenate([X, y.reshape((len(y), 1))], axis=1)



    X_INp = np.copy(X)
    X_mat = X_INp
    train_split = int(round(train_size * X_mat.shape[0]))
    train_data = X_mat[:train_split, :]
    x_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    x_test = X_mat[train_split:, :-1] 
    y_test = X_mat[train_split:, -1]

    return x_train, y_train, x_test, y_test






def plotCompare(x,ypred,yactual):

    plt.plot(x,yactual, color='r', label='actual')
    plt.plot(x,ypred, color='g', label='predicted')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x vs y")
    plt.legend()
    plt.show()



def plotxy(x,y):
    plt.plot(x,y, color='g', label='actual')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x vs y")
    plt.legend()
    plt.show()



def getAndCleanData():
    df = pd.read_csv('weatherAUS.csv')
    print(df.describe())
    print(df.info())

    # generating dummy variables
    df['RainTomorrow'] = df['RainTomorrow'].map({'Yes':1 ,'No':0})
    df['RainToday'] = df['RainToday'].map({'Yes':1 ,'No':0})
    df = df[['Rainfall', 'MinTemp', 'MaxTemp','WindGustSpeed','Temp9am','WindSpeed9am','Humidity9am','Pressure9am','Temp3pm','WindSpeed3pm','Humidity3pm','Pressure3pm','RainTomorrow','RainToday']]
    df = df.dropna()
    x = df[['Rainfall', 'MinTemp', 'MaxTemp','WindGustSpeed','Temp9am','WindSpeed9am','Humidity9am','Pressure9am','Temp3pm','WindSpeed3pm','Humidity3pm','Pressure3pm','RainToday']]

    y = df[['RainTomorrow']]
    # visualizeData(df)

    return x.to_numpy() ,y.to_numpy() 








def draw(totalSize):

    # generating sin testing data
    xdata = np.random.rand( totalSize)

    xdata = sorted(xdata)

    ydata = np.copy(xdata)
    for p in range(len(xdata)):
        ydata[p] = h(np.copy(xdata[p]))
   

    xout = [[1]]*len(xdata)


    xout = np.array(xout).astype('float64')
    for p in range(len(xdata)):
        xout[p][0] = xdata[p]
  
    return np.array(xout),np.array(ydata)


def visualizeData(df):
    # Checking for the class imbalance
    fig = plt.figure(figsize = (10, 6))
    axis = sns.countplot(x = 'RainTomorrow', data = df);
    axis.set_title('Class Distribution for the target feature', size = 16);
    for patch in axis.patches:
        axis.text(x = patch.get_x() + patch.get_width()/2, y = patch.get_height()/2, 
                s = f"{np.round(patch.get_height()/len(df)*100, 1)}%", 
                ha = 'center', size = 40, rotation = 0, weight = 'bold' ,color = 'white')
    axis.set_xlabel('Rain Tomorrow', size = 14)
    axis.set_ylabel('Count', size = 14);
    plt.show()



def saveModelWeightsAndScale(model,scale):
    with open('ModelAndScale.pickle', 'wb') as f:
        pickle.dump([model,scale], f, pickle.HIGHEST_PROTOCOL)

def openWeights():
    with open('ModelAndScale.pickle', 'rb') as f:
        x = pickle.load(f)
    return x[0],x[1]



def testSin():

    i = 5
    x,y = draw(400)
    x,y,xt,yt = load_data(x,y, train_size=0.9)
    scale = scaler(xt)
    xt = scale.scaleData(xt)


    plotxy(x,y)

    model = adamNN(i,40,1/10000)

    model.train(x,y)

    z = model.predictBased(x)
    plotCompare(x,z,y)




def trainAndScore():
    
 
    hiddenNodes  = 2 #selecting number fo hidden nodes
    batchSize = 16 #batch size
    epochs = 90
    learningRate = 1/10000



    # loading and cleaning data
    x,y = getAndCleanData()
    # splitting data
    x,y,xtest,ytest = load_data(x,y, train_size=0.75)
    #scaling data to prevent underflow when dividing by zero
    scale = scaler(x)
    x = scale.scaleData(x)
    xtest = scale.scaleData(xtest)




    print('Running model with '+str(hiddenNodes)+' hidden nodes')
    model = adamNN(hiddenNodes,batchSize,learningRate)
    start_time = time.time()
    tracemalloc.start()
    # training model
    model.train(x,y,epochs)

    totTime = time.time() - start_time
    print("The model took %s seconds to train" % (totTime))
    print("Maximum space used : "+str(tracemalloc.get_traced_memory()[1]))
    tracemalloc.stop()


    # making prediction on training dataset
    z = model.predictBasedInt(x)
    errorRate = error_rate(y,z)
    print('Training error rate = '+str(errorRate))

    print ('Accuracy Score :',accuracy_score(y, z)) 
    print ('Report : ')

    classification_report(y, z)


    # making prediction on testing dataset
    z = model.predictBasedInt(xtest)
    errorRate = error_rate(ytest,z)


    print('Testing error rate = '+str(errorRate))

    print ('Accuracy Score :',accuracy_score(ytest, z)) 
    print ('Report : ')

    classification_report(ytest, z)
    print()

    saveModelWeightsAndScale(model,scale)


def TestModels():
    x,y = getAndCleanData()

    # x,y,xtest,ytest = load_data(x,y, train_size=0.5)
    x,y,xtest,ytest = load_data(x,y, train_size=0.75)

    #scaling data to prevent underflow when dividing by zero
    model, scale = openWeights()




    #scaling data to prevent underflow when dividing by zero

    x = scale.scaleData(x)
    xtest = scale.scaleData(xtest)




 

    # making prediction on training dataset
    z = model.predictBasedInt(x)
    errorRate = error_rate(y,z)
    print('Training error rate = '+str(errorRate))

    print ('Accuracy Score :',accuracy_score(y, z)) 
    print ('Report : ')

    classification_report(y, z)




    # making prediction on testing dataset
    z = model.predictBasedInt(xtest)
    errorRate = error_rate(ytest,z)


    print('Testing error rate = '+str(errorRate))

    print ('Accuracy Score :',accuracy_score(ytest, z)) 
    print ('Report : ')

    classification_report(ytest, z)
    print()


    fig , ax = plt.subplots(1,2)

    fig.set_size_inches(12,4)
    ax[0].plot(model.trainAccuracy)
    ax[0].plot(model.valAccuracy)
    ax[0].set_title('Training Accuracy vs Validation Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='upper left')
    ax[1].plot(model.trainerror)
    ax[1].plot(model.valerror)
    ax[1].set_title('Training Error vs Validation Error')
    ax[1].set_ylabel('Error')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def main():

    # trainAndScore()
    TestModels()



if __name__ == "__main__":
    main()