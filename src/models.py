from keras import Input, backend as K, Model
from keras.constraints import MaxNorm
from keras.initializers import RandomUniform
from keras.layers import Embedding, Dropout, Lambda, Dense
from keras.optimizers import Adam

from src.embedding import DistMult
from src.evaluation import precision, recall, f1


def create_models(M, N, embedding_method):
    ed = 128
    kg_model = create_kg_model(N, M, ed, method=embedding_method)
    on_model = create_on_model(N, ed)
    metrics = {'score': ['acc', precision, recall, f1]}  # ,'x':['mae','mse',r2_keras]}
    metrics['x'] = metrics['score']
    metricsWithoutScore = {'x': ['acc', precision, recall, f1]}
    kg_model.compile(optimizer=Adam(lr=1e-3),
                     metrics=metrics,
                     loss={'score': 'binary_crossentropy', 'x': 'binary_crossentropy'},
                     loss_weights={'score': 1, 'x': 0.1})
    kg_model.summary()
    on_model.compile(optimizer=Adam(lr=1e-3),
                     metrics=metricsWithoutScore,
                     loss={'x': 'binary_crossentropy'},
                     loss_weights={'x': 1})
    on_model.summary()
    return kg_model, on_model


def create_kg_model(N, M, ed, dense_layers=(16, 16), method=DistMult):
    # kg model
    # embedding
    si, pi, oi = Input((1,)), Input((1,)), Input((1,))
    e = Embedding(N, ed, name='entity_embedding', embeddings_constraint=MaxNorm(1, axis=1))#, embeddings_initializer=RandomUniform(-0.05, 0.05))
    # Try to set the embeddings_initializer
    r = Embedding(M, ed, name='relation_embedding')
    s = Dropout(0.2)(e(si))
    p = Dropout(0.2)(r(pi))
    o = Dropout(0.2)(e(oi))
    score = method(s, p, o)
    # regression
    xi = Input((1,))
    x = Lambda(lambda x: K.squeeze(x, 1))(e(xi))
    for f in dense_layers:
        x = Dense(f, activation='relu')(x)
        x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid', name='x')(x)
    return Model(inputs=[si, pi, oi, xi], outputs=[score, x])


def create_on_model(N, ed, dense_layers=(16, 16)):
    # one-hot model
    # embedding
    e = Embedding(N, ed, name='entity_embedding', embeddings_constraint=MaxNorm(1, axis=1))
    # regression
    xi = Input((1,))
    x = Lambda(lambda x: K.squeeze(x, 1))(e(xi))
    for f in dense_layers:
        x = Dense(f, activation='relu')(x)
        x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid', name='x')(x)
    return Model(inputs=xi, outputs=x)

