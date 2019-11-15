### dense model

from keras import Model
from keras.layers import Dense, Dropout, BatchNormalization, Embedding, Multiply,Activation, Reshape, Concatenate, Lambda, Input, Flatten

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
from keras.constraints import MaxNorm
from tqdm import tqdm
import pandas as pd


def circ(s,o):
    s,o = tf.cast(s, dtype=tf.complex128),tf.cast(o,dtype=tf.complex128)
    return tf.real(tf.spectral.ifft(tf.conj(tf.spectral.fft(s))*tf.spectral.fft(o)))

def circular_cross_correlation(x, y):
    """Periodic correlation, implemented using the FFT.
    x and y must be of the same length.
    """
    return tf.real(tf.ifft(tf.multiply(tf.conj(tf.fft(tf.cast(x, tf.complex64))) , tf.fft(tf.cast(y, tf.complex64)))))

def HolE(s,p,o):
    # sigm(p^T (s \star o))
    # dot product in tf: sum(multiply(a, b) axis = 1)
    return tf.reduce_sum(tf.multiply(p, circular_cross_correlation(s, o)), axis = 1)

def TransE(s,p,o):
    return 1/tf.norm(s+p-o, axis = 1)


class LinkPredict(Model):

    def __init__(self,input_dim, embedding_dim = 128, use_bn=False, use_dp=False, embedding_method = 'DistMult'):
        super(LinkPredict, self).__init__(name='lp')
        self.use_bn = use_bn
        self.use_dp = use_dp
        
        if embedding_method == 'HolE':
            constraint = MaxNorm(1,axis=1)
        elif embedding_method == 'TransE':
            constraint = None
        elif embedding_method == 'DistMult':
            constraint = MaxNorm(1,axis=1)
        
        self.e = Embedding(input_dim[0],embedding_dim,embeddings_constraint=constraint)
        self.r = Embedding(input_dim[1],embedding_dim)
        
        if embedding_method == 'DistMult':
            self.embedding = [Multiply(),
                            Lambda(lambda x: K.sum(x, axis=-1)),
                            Activation('sigmoid'),
                            Reshape((1,))]
        elif embedding_method == 'HolE':
            self.embedding = [Lambda(lambda x: HolE(x[0],x[1],x[2])),
                            Activation('sigmoid'),
                            Reshape((1,))]
        elif embedding_method == 'TransE':
            self.embedding = [Lambda(lambda x: TransE(x[0],x[1],x[2])),
                              Activation('tanh'),
                              Reshape((1,))]
            
        else:
            raise NotImplementedError(embedding_method+' not implemented')
        
        self.rate = 0.2
        
        self.ls = [Dense(32, activation = 'relu'),
                   Dropout(self.rate),
                   Dense(1, activation='sigmoid'),
                   Reshape((1,))]
    
        if self.use_dp:
            self.dp = Dropout(self.rate)
            
        if self.use_bn:
            self.bn1 = BatchNormalization(axis=-1)
            self.bn2 = BatchNormalization(axis=-1)
            

    def call(self, inputs):
        # triple, chemical
        t,x = inputs
        
        s,p,o = t[:,0],t[:,1],t[:,2]
        
        s = self.e(s)
        p = self.r(p)
        o = self.e(o)
        
        x = self.e(x)
        
        if self.use_dp:
            s = self.dp(s)
            p = self.dp(p)
            o = self.dp(o)

        if self.use_bn:
            s = self.bn1(s)
            p = self.bn1(p)
            o = self.bn1(o)
            
        score = [s,p,o]
        
        for layer in self.embedding:
            score = layer(score)
        
        for layer in self.ls:
            x = layer(x)
        
        return score,x
    

def main(cv=False, verbose=2, file_number = 0, repeat = 0, ed=200):
    """
    cv: do cross validation.
    verbose: 0=silent, 1=batch, 2=epoch 
    file_number: file name results.
    repeat: repeat dataset if memory allows. Will reduce number of epochs.
    """

    Cg = read_data('./kg/chemical_graph.txt')
    Cg = pd.read_csv('./kg/kg.csv').dropna()
    Cg = list(zip(Cg['s'],Cg['p'],Cg['o']))
    dftr = pd.read_csv('./data/LC50_train.csv').dropna()
    dfte = pd.read_csv('./data/LC50_test.csv').dropna()
    Xtr,ytr = dftr['inchikey'], np.asarray(dftr['y']).reshape((-1,1))
    Xte,yte = dfte['inchikey'], np.asarray(dfte['y']).reshape((-1,1))

    C = set.union(*[set([a,b]) for a,_,b in Cg]) | set(Xtr) | set(Xte)
    Rc = set([r for _,r,_ in Cg])
    
    N, Nr = len(C), len(Rc)

    mapping_c = {c:i for c,i in zip(C,range(len(C)))}
    mapping_cr = {c:i for c,i in zip(Rc,range(len(Rc)))}
    
    #scaling and transforming continuous labels to binary.
    scaler = MinMaxScaler()
    Xtr = [mapping_c[c] for c in Xtr]
    ytr = np.around(scaler.fit_transform(ytr))
    Xte = np.asarray([mapping_c[c] for c in Xte]).reshape((-1,1))
    yte = np.around(scaler.transform(yte)).reshape((-1,))
    
    num_false = 10

    triples = []
    triple_labels = []
    
    C, Rc = list(C),list(Rc)

    for s,p,o in Cg:
        triples.append((mapping_c[s],mapping_cr[p],mapping_c[o]))
        triple_labels.append(1)
        
        for _ in range(num_false):
            triples.append((mapping_c[choice(C)],mapping_cr[p],mapping_c[choice(C)]))
            triple_labels.append(0)

    #Oversampling training
    Xtr,ytr = list(Xtr), list(ytr)
    u,c = np.unique(ytr, return_counts=True)
    while c[0] != c[1]:
        idx = np.random.choice(len(ytr))
        if ytr[idx] == u[np.argmin(c)]:
            ytr.append(ytr[idx])
            Xtr.append(Xtr[idx])
        u,c = np.unique(ytr, return_counts=True)
    
    triples, Xtr = np.asarray(triples),np.asarray(Xtr).reshape((-1,1))
    triple_labels, ytr = np.asarray(triple_labels).reshape((-1,)), np.asarray(ytr).reshape((-1,))

    # Equal length inputs
    if not cv:
        r = np.ceil(len(triples)/len(Xtr))
        if r > 1:
            Xtr = np.repeat(Xtr, r, axis=0)
            ytr = np.repeat(ytr, r, axis=0)
            Xtr = Xtr[:len(triples)]
            ytr = ytr[:len(triples)]
        else:
            triples = triples[:len(Xtr)]
            triple_labels = triple_labels[:len(Xtr)]
            

    losses = {"output_1": "binary_crossentropy",
            "output_2": "binary_crossentropy"
                }
    lW = {}
    lW['DistMult'] = {"output_1": 0.5, "output_2": 1.0}
    lW['HolE'] = {"output_1": 0.5, "output_2": 1.0}
    lW['TransE'] = {"output_1": 1.0, "output_2": 1.0}
    
    num_epochs = 100
    metrics = ['accuracy', keras_precision, keras_recall, keras_auc, keras_f1]
    callbacks = [EarlyStopping(monitor='output_2_loss', mode='min', patience=5, restore_best_weights=True)]
    results = {}

    for mode in ['TransE','DistMult','HolE']:
        lossWeights = lW[mode]
        
        if cv:
            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            cvscores = []
            with tqdm(total=10,desc='CV '+mode) as pbar:
                for train,test in kfold.split(Xtr,ytr):
                    model = LinkPredict(input_dim=[N,Nr], embedding_dim=ed, use_bn=True,use_dp=True, embedding_method=mode)
                    # Compile model
                    
                    model.compile(loss=losses, loss_weights=lossWeights, optimizer='adagrad', metrics=metrics, callbacks=callbacks)
                    # Fit the model
                    
                    x = Xtr[train]
                    y = ytr[train]
                    
                    tmpx = triples
                    tmpy = triple_labels
                    
                    m = max(len(tmpy),len(y))
                    while min(len(tmpy),len(y)) != m:
                        if len(tmpy) < m:
                            idx = np.random.choice(len(tmpy))
                            tmpx = np.concatenate((tmpx,tmpx[idx].reshape((1,3))),axis=0)
                            tmpy = np.append(tmpy,tmpy[idx])
                        if len(y) < m:
                            idx = np.random.choice(len(y))
                            x = np.append(x,x[idx])
                            y = np.append(y,y[idx])
                    
                    
                    x = x.reshape((-1,1))
    
                    model.fit([tmpx,x], [tmpy,y], epochs=num_epochs, batch_size=200, verbose=verbose, validation_split=0.0, shuffle=True)
                    
                    # evaluate the model
                    
                    scores = model.evaluate([triples[test],Xtr[test]], [triple_labels[test],ytr[test]], verbose=0, batch_size=len(ytr[test]))
                    
                    cvscores.append(scores)
                    
                    pbar.update(1)
            cvscores = np.asarray(cvscores)
            for i,n in enumerate(model.metrics_names):
                print(mode,n,"%.2f (+/- %.2f)" % (np.mean(cvscores[:,i]), np.std(cvscores[:,i])))
        
        else:
            model = LinkPredict(input_dim=[N,Nr], embedding_dim=ed, use_bn=True,use_dp=True, embedding_method=mode)
            model.compile('adagrad', loss=losses, loss_weights=lossWeights, metrics=metrics)
            
            bs = 1024
            
            model.fit([triples, Xtr],[triple_labels, ytr], epochs = num_epochs, shuffle=True, verbose=verbose, callbacks=callbacks, batch_size = bs,validation_split=0.0)

            r1 = model.evaluate([triples[:len(Xte)],Xte],[triple_labels[:len(yte)],yte], batch_size=200,verbose=0)
            results[mode] = r1
            
            if verbose in [1,2]:
                m = model.evaluate([triples[:len(Xte)],Xte],[triple_labels[:len(yte)],yte], batch_size=200,verbose=verbose)
                for a,b in zip(m,model.metrics_names):
                    print(b,a)
            
            p = model.predict([triples[:len(Xte)],Xte])[-1]
            with open('./results/LP/'+mode+'/'+str(file_number)+'.txt', 'w') as f:
                for a,b in zip(yte,p):
                    f.write(str(a)+','+str(b[-1])+'\n')
        
    return model, results

if __name__ == '__main__':
    main(cv=False,verbose=2,repeat=0,ed=16)
    #main(cv=True,verbose=0,repeat=0,ed=200)
    exit()
    num = 10
    cvscores = defaultdict(list)
    for i in tqdm(range(num)):
        model, scores = main(cv=False, verbose=0, file_number=i, repeat = 0)
        for k in scores:
            cvscores[k].append(scores[k])
        
    for k in cvscores:
        scores = np.asarray(cvscores[k])
        for i,n in enumerate(model.metrics_names):
            print(k, n,"%.2f (+/- %.2f)" % (np.mean(scores[:,i]), np.std(scores[:,i])))



















