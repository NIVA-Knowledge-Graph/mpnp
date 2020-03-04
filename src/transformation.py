from random import choice

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def generate_negative(kg, N, negative=2, check_kg=False):
    """Generates random negative triples
    to avoid all triples getting the same position"""
    true_kg = kg.copy()
    kg = []
    kgl = []
    for s, p, o, score in true_kg:
        kg.append((s, p, o))
        kgl.append(score)
        for _ in range(negative):
            t = (choice(range(N)), p, choice(range(N)))

            if check_kg:
                if not t in true_kg:
                    kg.append(t)
                    kgl.append(0)
            else:
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


def map_entities_and_relations(Xte, Xtr, kg, entities, relations):
    """Gives all URIs a corresponding integer id"""
    me = {e: i for i, e in enumerate(entities)}
    mr = {e: i for i, e in enumerate(relations)}
    true_kg = [(me[s], mr[p], me[o], score) for s, p, o, score in kg]
    Xtr = [me[s] for s in Xtr]
    Xte = [me[s] for s in Xte]
    return Xte, Xtr, true_kg


def contained_in_kg_filter(Xte, Xtr, yte, ytr, entities):
    """Filters out train and test data thats not contained in the knowledge graph"""
    tr = [(x, y) for x, y in zip(Xtr, ytr) if x in entities]
    te = [(x, y) for x, y in zip(Xte, yte) if x in entities]
    Xtr, ytr = zip(*tr)
    Xte, yte = zip(*te)
    Xtr, ytr = list(Xtr), np.asarray(ytr).reshape((-1, 1))
    Xte, yte = list(Xte), np.asarray(yte).reshape((-1, 1))
    return Xte, Xtr, yte, ytr


def convert_to_binary(yte, ytr):
    """devides the dataset on the median
    and gives it a binary value based on with pole its in"""
    scaler = MinMaxScaler()
    ytr = scaler.fit_transform(ytr)
    yte = scaler.transform(yte)
    ytr = np.abs(np.around(ytr - np.median(ytr) + 0.5) - 1)
    yte = np.abs(np.around(yte - np.median(yte) + 0.5) - 1)
    ytr = list(ytr.reshape((-1,)))
    yte = list(yte.reshape((-1,)))
    return yte, ytr