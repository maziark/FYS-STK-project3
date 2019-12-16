## -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Dropout, Subtract
from sklearn.model_selection import train_test_split
from keras import backend as K
import matplotlib.pyplot as plt

def normalize(x):
    #Function to normalize data to lie in [0,1].
    return (x-np.amin(x))/(np.amax(x)-np.amin(x))

def acc(y_true, y_pred):
    #Accuracy function for use with siames twin network
    ones = K.ones_like(y_pred)
    return K.mean(
            K.equal(y_true, ones - K.clip(K.round(y_pred), 0, 1)), axis=-1)

def eucl_dist_output_shape(shapes):
    #Helper function to assure correct network shapes
    shape1, shape2 = shapes
    return (shape1[0], 1)

def euclidean_distance(vectors):
    #Calculates distance between imbedded tensors
    x, y = vectors
    sum_square = K.sum(K.square(x-y),axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))
       
def contrastive_loss(y_true, y_pred):
    #This is the contrastive loss from Hadsell et al. 
    #If y_true= 1, meaning that the inputs come from the same category
    #then the square of y_pred is returned and the network
    #will try to minimize y_pred
    #If y_true=0, meaning different categories, the function
    #returns, if the difference beween margin and ypred > 0,
    #the square of this difference. Thus the network will try to maximize
    #y_pred
    margin = 0.9
    square_pred = K.square(y_pred)
    margin_square = K.square((K.maximum(margin - y_pred, 0)))
    return K.mean(y_true*square_pred + (1-y_true)*margin_square)

def triplet_loss(y_true,y_pred):
    #Function is FaceNets triplet loss. Y_pred is the difference
    #between the positive distance layer and the negative distance layer
    margin = 0.9
    return K.mean(K.maximum(y_pred+margin,0))

def triplet_acc(y_true,y_pred):
    return K.mean(K.less(y_pred,0))
 
def set_category(data,train=None,test=None):
    #The creation of bins/categories  aims to create to categories from
    #the data set, seperated by the median. The digitize function
    #of numpy returns an array with 1,2,3....n as labels for each of n 
    #bins. The min, max cut off are  chosen to be larger/smaller than 
    #max min values of the data
    bins = np.array([0,data.median(),100])
    if train.empty:
        temp = np.digitize(data,bins)
        return temp-1
    train_labels = np.digitize(train,bins)
    test_labels = np.digitize(test,bins)
    return train_labels-1, test_labels-1

def make_anchored_pairs(data,target,test_data,test_target,anch):
    '''
    This function returns anchored of data points for comparison
    in siamese oneshot twin network
    '''
    #create lsits to store pairs and labels to return
    pairs = []
    labels = []
    test_pairs = []
    test_labels = []
    #Create all possible training pairs where one element is an anchor
    for i, a in enumerate(anch):
        x1 = a
        for index in range(len(data)):
            x2 = data[index]
            labels += [1] if target[index] == i else [0]
            pairs += [[x1,x2]]
            print(len(labels))
        for index in range(len(test_data)):
            x2 = test_data[index]
            test_labels += [1] if test_target[index] == i else [0]
            test_pairs += [[x1,x2]]
            
    return np.array(pairs),np.array(labels),np.array(test_pairs),np.array(
            test_labels)

def make_training_triplets(anchors,samples,labels):
    #Creates triplets for siamese triplet network.
    triplets = []
    for s,l in zip(samples,labels):
        positive = anchors[0] if l==0 else anchors[1]
        negative = anchors[0] if l==1 else anchors[1]
        triplets += [[positive,negative,s]]
    return np.array(triplets)
        
def base_network(input_shape):
    #A low complexity base network. Duplicated for siamese networks.
    base_input = Input(shape=input_shape)
    x = Dense(50,activation="relu")(base_input)
    x = Dropout(0.1)(x)
    x = Dense(50,activation="relu")(base_input)
    x = Dropout(0.1)(x)
    return Model(base_input,x)

def complex_base_network(input_shape):
    #A high complexity base network. Duplicated for siamese networks.
    base_input = Input(shape=input_shape)
    x = Dense(100,activation="relu")(base_input)
    x = Dropout(0.1)(x)
    x = Dense(100,activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(100,activation="relu")(x)
    x = Dropout(0.1)(x)
    return Model(base_input,x)

def triplet(t_train,t_test,train_labels,test_labels):
    base_model = complex_base_network((input_size,))
    
    input1 = Input(shape=(input_size,))
    input2 = Input(shape=(input_size,))
    input3 = Input(shape=(input_size,))
    
    pos = base_model(input1)
    neg = base_model(input2)
    sam = base_model(input3)
    
    #Check distance between positive and sample
    merge_layer1 = Lambda(
            euclidean_distance,output_shape=eucl_dist_output_shape)(
            [pos,sam])
    #Check distance between negative and sample
    merge_layer2 = Lambda(
            euclidean_distance,output_shape=eucl_dist_output_shape)(
            [neg,sam])
    #Compare distances
    loss_layer = Subtract()([merge_layer1,merge_layer2])
    siamese_model = Model(inputs=[input1,input2,input3],outputs=loss_layer)    
    siamese_model.compile(loss=triplet_loss,
                    optimizer="Adam",metrics=[triplet_acc])    
    siamese_model.summary()    
    history = siamese_model.fit([t_train[:,0],t_train[:,1],t_train[:,2]],
                train_labels[:],validation_data=(
                        [t_test[:,0],t_test[:,1],t_test[:,2]],test_labels[:]),
                        batch_size=5,epochs=200)
    return history
    
def simple(pairs_train,labels_train,pairs_test,labels_test):
    base_model = base_network((input_size,))
    
    input1 = Input(shape=(input_size,))
    input2 = Input(shape=(input_size,))
    
    run1 = base_model(input1)
    run2 = base_model(input2)
    
    #Check distance between anchor and sample
    merge_layer = Lambda(
            euclidean_distance,output_shape=eucl_dist_output_shape)(
                    [run1,run2])
    out_layer = Dense(1,activation="sigmoid")(merge_layer)
    siamese_model = Model(inputs=[input1,input2],outputs=out_layer)    
    siamese_model.compile(loss=contrastive_loss,
                    optimizer="Adam",metrics=[acc])    
    siamese_model.summary()    
    history = siamese_model.fit([pairs_train[:,0],pairs_train[:,1]],
                labels_train[:],validation_data=(
                        [pairs_test[:,0],pairs_test[:,1]],labels_test[:]),
                        batch_size=5,epochs=200)
    return history

def compl(pairs_train,labels_train,pairs_test,labels_test):
    base_model = complex_base_network((input_size,))
    
    input1 = Input(shape=(input_size,))
    input2 = Input(shape=(input_size,))
    
    run1 = base_model(input1)
    run2 = base_model(input2)
    
    merge_layer = Lambda(
            euclidean_distance,output_shape=eucl_dist_output_shape)(
                    [run1,run2])
    out_layer = Dense(1,activation="sigmoid")(merge_layer)
    siamese_model = Model(inputs=[input1,input2],outputs=out_layer)    
    siamese_model.compile(loss=contrastive_loss,
                    optimizer="Adam",metrics=[acc])
    siamese_model.summary()    
    history = siamese_model.fit([pairs_train[:,0],pairs_train[:,1]],
                labels_train[:],validation_data=(
                        [pairs_test[:,0],pairs_test[:,1]],labels_test[:]),
                batch_size=5,epochs=200)
    return history

def visualize(history1,history2,history3,seed,save=False):
    #Function creates plot based on keras history objects
    #will save plots if save=True
    fig, (ax1, ax2,ax3) = plt.subplots(3, 1,figsize=(6, 10))    
    ax1.plot(history1.history['acc'])
    ax1.plot(history1.history['val_acc'])
    ax1.set_title("Simple Model")
    ax2.plot(history2.history['acc'])
    ax2.plot(history2.history['val_acc'])
    ax2.set_title("Complex Model")
    ax3.plot(history3.history['triplet_acc'])
    ax3.plot(history3.history['val_triplet_acc'])
    ax3.set_title("Triplet Model")
    fig.suptitle('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    if save: fig.savefig("{}plotaccSNN{}.png".format(path,seed))
    
    fig, (ax1, ax2,ax3) = plt.subplots(3, 1,figsize=(6, 10))    
    ax1.plot(history1.history['loss'])
    ax1.plot(history1.history['val_loss'])
    ax1.set_title("Simple Model")
    ax2.plot(history2.history['loss'])
    ax2.plot(history2.history['val_loss'])
    ax2.set_title("Complex Model")
    ax3.plot(history3.history['loss'])
    ax3.plot(history3.history['val_loss'])
    ax3.set_title("Triplet Model")
    fig.suptitle('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    if save: fig.savefig("{}plotlossSNN{}.png".format(path,seed))
    

    
path = "e:\\Data\\Skole\\Project3\\"
#Imports a table containing DNA sequence counts, a proxy for species abundance
#for over 15000 species in 72 different lakes
data = pd.read_csv("{}ASV_table_mod.csv".format(path))

#Imports various abiotic and biotic data from the 72 lakes.
#Variables of interest are TP=Total phosphorus, Temperature, N2O amount,
#pH, 
meta = pd.read_csv("{}Metadata_table.tsv".format(path),delimiter=r"\s+")
#Remove columns with 0s. I.E. species that are not present at all
#sites.
dt = data.replace(0, pd.np.nan).dropna(axis=1, how='any').fillna(0).astype(int)
#Normalize data   
data = normalize(dt)
ph = meta["pH"]
n2o = meta["N2O"]
temp = meta["Temperature"]
tp = meta["TP"]
#Anchors selected by inspection. Serves as positive/negatives for
#comparison in network. First position is a low valued representative
#second position is a high valued representative.
anchors = {"TMP":(data.iloc[10],data.iloc[70])}
anchors["TP"] = (data.iloc[15],data.iloc[67])
#Convert df to array
X = data.to_numpy()
#Set input size for network
input_size = len(X[0,:])
for run in range(5):
    seed = run      
    target_choice = "TP"
    np.random.seed(seed)
    
    y = meta[target_choice]
    
    #y = to_categorical(y, num_classes=None)
    #Split into training and testing
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    #Creates category labels
    y_train_l,y_test_l = set_category(y,y_train,y_test)
    #Make pairs and labels of "same" or "differen"
    pairs_train, labels_train, pairs_test, labels_test = make_anchored_pairs(
            X_train,y_train_l,X_test,y_test_l,anch=anchors[target_choice])
    #Create triplets for triplet network
    triplets_train = make_training_triplets(
            anchors[target_choice],X_train,y_train_l)
    triplets_test = make_training_triplets(
            anchors[target_choice],X_test,y_test_l)
    
    #Set input size for network
    input_size = len(X[0,:])
    simp = simple(pairs_train,labels_train,pairs_test,labels_test)
    comp = compl(pairs_train,labels_train,pairs_test,labels_test)
    trip = triplet(triplets_train,triplets_test,y_train_l,y_test_l)
    visualize(simp,comp,trip,seed,save=False)
