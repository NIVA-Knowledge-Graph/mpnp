### dense model

from keras import Model
from keras.layers import Dense, Dropout, BatchNormalization, Embedding, Multiply,Activation, Reshape, Concatenate, Lambda, Input, Flatten, Add

from keras.utils import plot_model

import numpy as np
from random import choice
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from metrics import keras_auc,keras_precision,keras_recall,keras_f1,keras_fb
from common import read_data
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping
from collections import defaultdict
from keras.constraints import MaxNorm, UnitNorm
from tqdm import tqdm
import pandas as pd
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2,l1,l1_l2
from tqdm import tqdm

from keras.optimizers import Adam


def circular_cross_correlation(x, y):
    """Periodic correlation, implemented using the FFT.
    x and y must be of the same length.
    """
    return tf.real(tf.signal.ifft(tf.multiply(tf.math.conj(tf.signal.fft(tf.cast(x, tf.complex64))) , tf.signal.fft(tf.cast(y, tf.complex64)))))

def HolE(s,p,o):
    # sigm(p^T (s \star o))
    # dot product in tf: sum(multiply(a, b) axis = 1)
    score = tf.reduce_sum(tf.multiply(p, circular_cross_correlation(s, o)), axis = -1)
    score = Activation('sigmoid', name='score')(score)
    return score

def TransE(s,p,o):
    score = Add()([s,p,-o])
    score = tf.norm(score,axis=-1)
    return Activation('tanh', name='score')(1/score)
    
def DistMult(s,p,o):
    score = Multiply()([s,p,o])
    score = Lambda(lambda x: K.sum(x, axis=-1))(score)
    return Activation('sigmoid', name='score')(score)



def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    r = true_positives / (possible_positives + K.epsilon())
    return r

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    p = true_positives / (predicted_positives + K.epsilon())
    return p

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*((p*r)/(p+r+K.epsilon()))

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def generate_negative(kg, N, negative = 2, check_kg = False):
    #false triples:
    true_kg = kg.copy()
    kg = []
    kgl = []
    for s,p,o in true_kg:
        kg.append((s,p,o))
        kgl.append(1)
        for _ in range(negative):
            t = (choice(range(N)), p, choice(range(N)))
            if check_kg and not t in true_kg:
                kg.append(t)
                kgl.append(0)
            
    return kg, kgl
    
def balance_inputs(input1, input1l, input2, input2l):
    input1 = list(input1)
    input2 = list(input2)
    input1l = list(input1l)
    input2l = list(input2l)
    
    while len(input1) != len(input2):
        if len(input1) < len(input2):
            idx = choice(range(len(input1)))
            input1.append(input1[idx])
            input1l.append(input1l[idx])
            
        if len(input1) > len(input2):
            idx = choice(range(len(input2)))
            input2.append(input2[idx])
            input2l.append(input2l[idx])
            
    return input1, input1l, input2, input2l

def create_kg_model(N, M, ed, dense_layers = (16,16)):
    #kge model
    #embedding
    si,pi,oi = Input((1,)), Input((1,)), Input((1,))
    e = Embedding(N, ed, embeddings_constraint=MaxNorm(axis=1), name='entity_embedding')
    r = Embedding(M, ed, name='relation_embedding')
    s = Dropout(0.2)(e(si))
    p = Dropout(0.2)(r(pi))
    o = Dropout(0.2)(e(oi))
    score = DistMult(s,p,o)
    
    #regression
    xi = Input((1,))
    x = Lambda(lambda x: K.squeeze(x, 1))(e(xi))
    for f in dense_layers:
        x = Dense(f,activation=LeakyReLU(0.1))(x)
        x = Dropout(0.2)(x)
    x = Dense(1,activation='sigmoid', name='x')(x)
    
    return Model(inputs=[si,pi,oi,xi], outputs=[score,x])

def create_on_model(N, ed, dense_layers = (16,16)):
    #one-hot model
    #embedding
    e = Embedding(N, ed, embeddings_constraint=MaxNorm(axis=1),name='entity_embedding')
    
    #regression
    xi = Input((1,))
    x = Lambda(lambda x: K.squeeze(x, 1))(e(xi))
    for f in dense_layers:
        x = Dense(f,activation=LeakyReLU(0.1))(x)
        x = Dropout(0.2)(x)
    x = Dense(1,activation='sigmoid', name='x')(x)
    
    return Model(inputs=xi, outputs=x)

def main():
    
    kg = pd.read_csv('./kg/kg_chebi.csv')
    kg = list(zip(kg['s'],kg['p'],kg['o']))
    entities = set([s for s,p,o in kg]) | set([o for s,p,o in kg])
    relations = set([p for s,p,o in kg])
    
    dftr = pd.read_csv('./data/LC50_train.csv').dropna()
    dfte = pd.read_csv('./data/LC50_test.csv').dropna()
    Xtr,ytr = list(dftr['inchikey']), list(dftr['y'])
    Xte,yte = list(dfte['inchikey']), list(dfte['y'])
    
    tr = [(x,y) for x,y in zip(Xtr,ytr) if x in entities]
    te = [(x,y) for x,y in zip(Xte,yte) if x in entities]
    
    Xtr,ytr = zip(*tr)
    Xte,yte = zip(*te)
    Xtr, ytr = list(Xtr), np.asarray(ytr).reshape((-1,1))
    Xte, yte = list(Xte), np.asarray(yte).reshape((-1,1))
    
    #convert to binary
    scaler = MinMaxScaler()
    ytr = scaler.fit_transform(ytr)
    yte = scaler.transform(yte)
    ytr = np.around(ytr-np.median(ytr)+0.5)
    yte = np.around(yte-np.median(yte)+0.5)
    
    ytr = list(ytr.reshape((-1,)))
    yte = list(yte.reshape((-1,)))
    
    l1 = len((set(Xtr) | set(Xte)) - entities)
    print('Proportion of entites missing in KG:',l1/len(set(Xtr) | set(Xte)))
    
    entities = list(entities)
    relations = list(relations)
    
    me = {e:i for i,e in enumerate(entities)}
    mr = {e:i for i,e in enumerate(relations)}
    
    #mapping 
    true_kg = [(me[s],mr[p],me[o]) for s,p,o in kg]
    Xtr = [me[s] for s in Xtr]
    Xte = [me[s] for s in Xte]
    
    #modelling
    N = len(entities)
    M = len(relations)
    ed = 64
    
    kg_model = create_kg_model(N,M,ed)
    on_model = create_on_model(N,ed)
    
    metrics = {'score':['acc',precision,recall,f1]}#,'x':['mae','mse',r2_keras]}
    metrics['x'] = metrics['score']
    epochs = 100
    
    kg_model.compile(optimizer=Adam(lr=1e-3, decay=1e-3 / epochs), metrics=metrics,loss={'score':'binary_crossentropy','x':'binary_crossentropy'}, loss_weights={'score':1,'x':0.5})
    kg_model.summary()
    
    on_model.compile(optimizer=Adam(lr=1e-3, decay=1e-3 / epochs), metrics=metrics,loss={'x':'binary_crossentropy'}, loss_weights={'x':1})
    on_model.summary()
    
    for i in tqdm(range(epochs)):
        kg, kgl = generate_negative(true_kg, N, negative = 4, check_kg = False)
        kg, kgl, tmpXtr, tmpytr = balance_inputs(kg,kgl,Xtr,ytr)
    
        X1 = np.asarray(kg)
        y1 = np.asarray(kgl).reshape((-1,))
        X2 = np.asarray(tmpXtr)
        y2 = np.asarray(tmpytr).reshape((-1,))
        
        inputs = [X1[:,0], X1[:,1], X1[:,2], X2]
        outputs = [y1, y2]
        
        if i/epochs > 0.75:
            kg_model.get_layer('entity_embedding').trainable=False
            kg_model.get_layer('relation_embedding').trainable=False
            on_model.get_layer('entity_embedding').trainable=False
            
        # train the model
        kg_model.fit(
            inputs, outputs,
            epochs=1, batch_size=2**11, verbose=1)
        
        on_model.fit(
            X2, y2,
            epochs=1, batch_size=2**11, verbose=1)
    
    
    X1 = np.asarray(kg[:len(yte)])
    y1 = np.asarray(kgl[:len(yte)]).reshape((-1,))
    X2 = np.asarray(Xte)
    y2 = np.asarray(yte).reshape((-1,))
    
    inputs = [X1[:,0], X1[:,1], X1[:,2], X2]
    outputs = [y1, y2]
    
    e1 = kg_model.evaluate(inputs, outputs, batch_size=2**11)
    e2 = on_model.evaluate(X2, y2, batch_size=2**11)

    d = defaultdict(list)
    
    for n,v in zip(kg_model.metrics_names,e1):
        d[n].append(v)
    for n,v in zip(on_model.metrics_names,e2):
        d['x_'+n].append(v)
        
    for k in d:
        print(k,d[k])
        
    tmp = list(np.unique(yte,return_counts=True)[-1]/len(yte))
    print('Prior', max(tmp))

if __name__ == '__main__':
    main()
    

















