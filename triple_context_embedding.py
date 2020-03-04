import tensorflow as tf


def get_neighbor(h):
    # use map to look up
    return  # neighbors {{r:1,t:2},{r:1,t:2}}


def get_relations(h, t):
    return  # list of relations it have to traverse trough {1,2,3,4}


def tce(h, r, t):
    # use map to look up
    return f1 * f2 * f3


def score_transe(s, p, o):
    return 1 / tf.norm(s + p - o, axis=1)


def score_path(h, relation_path, t):
    p = []
    for path in relation_path:
        p.append(sum(path))
    return 1 / tf.norm(h + p - t, axis=1)


def f1(h):
    neighbors = get_neighbor(h)

    if not neighbors:
        return -100
    s = 0.0
    for neighbor in neighbors:
        s += score_transe(h, neighbor['r'], neighbor['t'])
    return (0 - s) / len(neighbors)


def f2(h, t):
    s = 0.0
    relations = get_relations(h, t)
    if not relations:
        return -100

    for relation in relations:
        s += score_path(h, relation, t)
    return (0 - s) / len(relation)


def f3(h, r, t):
    return 0 - score_transe(h, r, t)
