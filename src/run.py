import numpy as np

from collections import defaultdict
import pandas as pd
from keras.callbacks import EarlyStopping
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from src.models import create_models
from src.transformation import generate_negative, balance_inputs, map_entities_and_relations, \
    contained_in_kg_filter, convert_to_binary
from src.embedding import DistMult, TransE
from src.file_management import add_result, read_train_and_test_data


def main():
    """ To group together similar runs, they should have the same prefix.

    The prefix should be on this form:
    [name_of_the_method]_[avg_or_normalized_if_not_binary]_[embedding_method]_[uniq_run_id]
    E.G. directed_one_step_back_avg_distmult_1

    This way its easier to do statistics
    """

    #run_full_kg('7', TransE, epochs=100)


    # files = ('./kg/sorted_kg_w_touched_two_Steps.csv',
    #          './kg/sorted_kg_w_touched_directed_one_step_back.csv')
    # names = ('two_step',
    #          'directed_one_step_back')
    # for file, name in zip(files, names):
    #
    #     for i in range(7):
    #         print('run: ' + str(i) + ', savenameprefix: ' + name + ' filename' + file)
    #         run_scored_kg_avg(file, 'es_' + name + '_avg_TransE_' + str(i), TransE)
    #         run_scored_kg_normalized(file, 'es_' + name + '_normalized_TransE_' + str(i), TransE)
    #         run_scored_kg_avg(file, 'es_' + name + '_avg_distmult_' + str(i), DistMult)
    #         run_scored_kg_normalized(file, 'es_' + name + '_normalized_distmult_' + str(i), DistMult)

    # es_two_step_normalized_TransE
    # es_two_step_normalized_distmult
    # es_two_step_avg_TransE
    # es_two_step_avg_distmult

    # es_directed_one_step_back_normalized_TransE
    # es_directed_one_step_back_normalized_distmult
    # es_directed_one_step_back_avg_TransE
    # es_directed_one_step_back_avg_distmult


    # es_descending_influence_normalized_TransE
    # es_descending_influence_normalized_distmult
    # es_descending_influence_avg_TransE
    # es_descending_influence_avg_distmult

    # decending influence - manual
    # './kg/sorted_kg_w_touched_descending_influence.csv'
    # 'descending_influence'
    for i in range(7):
        print('run: ' + str(i) + ', savenameprefix: ' + 'descending_influence' + ' filename ' + './kg/sorted_kg_w_touched_descending_influence.csv')
        run_scored_kg_normalized('./kg/sorted_kg_w_touched_descending_influence_avg.csv', 'es_' + 'descending_influence' + '_avg_TransE_' + str(i), TransE)
        run_scored_kg_normalized('./kg/sorted_kg_w_touched_descending_influence.csv', 'es_' + 'descending_influence' + '_normalized_TransE_' + str(i), TransE)
        run_scored_kg_normalized('./kg/sorted_kg_w_touched_descending_influence_avg.csv', 'es_' + 'descending_influence' + '_avg_distmult_' + str(i), DistMult)
        run_scored_kg_normalized('./kg/sorted_kg_w_touched_descending_influence.csv', 'es_' + 'descending_influence' + '_normalized_distmult_' + str(i), DistMult)
    # --------------------------

    # run_scored_kg_avg('./kg/sorted_kg_w_touched_two_Steps.csv', 'two_step_avg_TransE_7', TransE) # 4, 6! bad

    # run_scored_kg_normalized('./kg/sorted_kg_w_touched_two_Steps.csv', 'two_step_normalized_TransE_7', TransE) # 2 bad, 3? 5 bad

    # ---------------------------

    # run_scored_kg_normalized('./kg/sorted_kg_w_touched_directed_one_step_back.csv',
    #                         'directed_one_step_back_normalized_TransE_7',
    #                         TransE)

    # --directed_one_step_back_avg_TransE_2--

    # run_scored_kg_avg('./kg/sorted_kg_w_touched_directed_one_step_back.csv',
    #                  'directed_one_step_back_avg_TransE_7',
    #                  TransE)

    # -----------------

    # run_scored_kg_normalized('./kg/sorted_kg_w_touched_descending_influence.csv',
    #                         'descending_influence_normalized_TransE_7',
    #                         TransE)

    #run_scored_kg_normalized('./kg/sorted_kg_w_touched_descending_influence_avg.csv',
    #                  'descending_influence_avg_TransE_7',
    #                  TransE)

    # -----------------


def run_scored_kg_normalized(filename, save_name, embedding_method, epochs=100):
    """runs with the score normalized between 0.5 and 1"""
    kg = pd.read_csv(filename)
    kg = list(zip(kg['s'], kg['p'], kg['o'], kg['score']))
    max_score = max([score for s, p, o, score in kg])
    kg = [(s, p, o, score / (max_score * 2) + 0.5) for s, p, o, score in kg]
    run(kg, save_name, epochs, embedding_method)


def run_scored_kg_avg(filename, save_name, embedding_method, epochs=100):
    """runs with the average of of the score before normalizing it"""
    kg = pd.read_csv(filename)
    kg = list(zip(kg['s'], kg['p'], kg['o'], kg['score'], kg['touched']))
    kg = [(s, p, o, score / touched if touched != 0 else score) for s, p, o, score, touched in kg]
    max_score = max([score for s, p, o, score in kg])
    kg = [(s, p, o, score / (max_score * 2) + 0.5) for s, p, o, score in kg]
    run(kg, save_name, epochs, embedding_method)


def run_full_kg(run_id, embedding_method, epochs=100):
    """Runs with the whole KG with binary scores"""
    # save_name = 'test_epoc_' + str(epochs) + '_full_kg_distmult_' + run_id
    embedding_str = 'unknown'
    if embedding_method == DistMult:
        embedding_str = 'distmult'
    elif embedding_method == TransE:
        embedding_str = 'TransE'
    save_name = 'es_full_kg_' + embedding_str + '_' + run_id
    kg = pd.read_csv('./kg/kg_chebi_CID.csv')
    kg['score'] = 1
    kg = list(zip(kg['s'], kg['p'], kg['o'], kg['score']))
    kgmesh = pd.read_csv('./kg/kg_mesh_CID.csv')
    kgmesh['score'] = 1
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o'], kgmesh['score']))
    kg = kg + kgmesh
    run(kg, save_name, epochs, DistMult)


def run_only_scored_kg(run_id, epochs=100):
    """Runs with only the scored triples with binary scores"""
    save_name = 'scored_kg_distmult_' + run_id
    kg = pd.read_csv('./kg/sorted_kg.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o'], kg['score']))
    kg = [(s, p, o) for s, p, o, score in kg if score > 0]
    run(kg, save_name, epochs, DistMult)


def run_only_cid_mapped_kg(run_id, epochs=100):
    """Runs with only the cid mapped triples with binary scores"""
    save_name = 'es_only_cid_mapped_distmult_' + run_id
    kg = pd.read_csv('./kg/only_cid_mapped_kg.csv')
    kg['score'] = 1
    kg = list(zip(kg['s'], kg['p'], kg['o'], kg['score']))
    run(kg, save_name, epochs, DistMult)


def run_only_cid_mapped_kg_transe(run_id, epochs=100):
    """Runs with only the cid mapped triples with binary scores"""
    save_name = 'es_only_cid_mapped_TransE_' + run_id
    kg = pd.read_csv('./kg/only_cid_mapped_kg.csv')
    kg['score'] = 1
    kg = list(zip(kg['s'], kg['p'], kg['o'], kg['score']))
    run(kg, save_name, epochs, TransE)


def run(kg, result_name, epochs, embedding_method):
    entities = set([s for s, p, o, score in kg]) | set([o for s, p, o, score in kg])
    relations = set([p for s, p, o, score in kg])

    Xte, Xtr, yte, ytr = read_train_and_test_data()

    missing_entities = len((set(Xtr) | set(Xte)) - entities)
    print('Proportion of entites missing in KG:', missing_entities / len(set(Xtr) | set(Xte)))

    Xte, Xtr, yte, ytr = contained_in_kg_filter(Xte, Xtr, yte, ytr, entities)
    yte, ytr = convert_to_binary(yte, ytr)
    entities = list(entities)
    relations = list(relations)
    Xte, Xtr, true_kg = map_entities_and_relations(Xte, Xtr, kg, entities, relations)

    # modelling
    N = len(entities)
    M = len(relations)
    kg_model, on_model = create_models(M, N, embedding_method)

    bs = 2 ** 20

    best_loss1 = 0 #np.Inf
    best_loss2 = 0 #np.Inf

    # stop training params
    patience = 5
    patience_delta = 0.001
    p1 = patience
    p2 = patience

    losses = []

    for i in tqdm(range(epochs)):
        if p1 <= 0 and p2 <= 0: break

        kg, kgl = generate_negative(true_kg, N, negative=4, check_kg=False)
        kg, kgl, tmpXtr, tmpytr = balance_inputs(kg, kgl, Xtr, ytr)

        X1 = np.asarray(kg)
        y1 = np.asarray(kgl).reshape((-1,))
        X2 = np.asarray(tmpXtr)
        y2 = np.asarray(tmpytr).reshape((-1,))

        inputs = [X1[:, 0], X1[:, 1], X1[:, 2], X2]
        outputs = [y1, y2]

        # warmup_embedding(i, epochs, kg_model)

        # freeze embeddings toward end of training.
        if i / epochs >= 0.9:
            kg_model.get_layer('entity_embedding').trainable = False
            kg_model.get_layer('relation_embedding').trainable = False
            on_model.get_layer('entity_embedding').trainable = False

        if p1 > 0:
            l1 = kg_model.fit(
                inputs, outputs,
                epochs=1, batch_size=bs, verbose=1)
            loss1 = l1.history['loss'][-1]

        if p2 > 0:
            l2 = on_model.fit(
                X2, y2,
                epochs=1, batch_size=bs, verbose=1)
            loss2 = l2.history['loss'][-1]

        if loss1 <= best_loss1 - patience_delta:
            p1 -= 1

        if loss2 <= best_loss2 - patience_delta:
            p2 -= 1

        if loss1 > best_loss1:
            best_loss1 = loss1
            p1 = patience

        if loss2 > best_loss2:
            best_loss2 = loss2
            p2 = patience

        losses.append((l1.history['loss'], l1.history['score_loss'], l1.history['x_loss'], l2.history['loss']))

    # plot_loss(losses)

    d = evaluate(Xte, yte, bs, kg, kgl, on_model, kg_model)
    add_result(result_name, d)


def evaluate(Xte, yte, bs, kg, kgl, on_model, kg_model):
    """evaluates both models and return the dict with the values"""
    X1 = np.asarray(kg[:len(yte)])
    y1 = np.asarray(kgl[:len(yte)]).reshape((-1,))
    X2 = np.asarray(Xte)
    y2 = np.asarray(yte).reshape((-1,))
    inputs = [X1[:, 0], X1[:, 1], X1[:, 2], X2]
    outputs = [y1, y2]

    d = defaultdict(list)

    e1 = kg_model.evaluate(inputs, outputs, batch_size=bs)
    y_pred_kg = kg_model.predict(inputs)[1].ravel()
    fpr_kg, tpr_kg, thresholds_kg = roc_curve(outputs[-1], y_pred_kg)
    auc_kg = auc(fpr_kg, tpr_kg)
    d['KG auc'].append(auc_kg)
    for n, v in zip(kg_model.metrics_names, e1):
        d['KG ' + n].append(v)

    e2 = on_model.evaluate(inputs[-1], outputs[-1], batch_size=bs)
    y_pred_on = on_model.predict(inputs[-1]).ravel()
    fpr_on, tpr_on, thresholds_on = roc_curve(outputs[-1], y_pred_on)
    auc_on = auc(fpr_on, tpr_on)
    d['One-Hot auc'].append(auc_on)
    for n, v in zip(on_model.metrics_names, e2):
        d['One-Hot ' + n].append(v)

    for k in d:
        print(k, d[k])

    d['One-Hot fpr'].append(fpr_on)
    d['One-Hot tpr'].append(tpr_on)
    d['KG fpr'].append(fpr_kg)
    d['KG tpr'].append(tpr_kg)

    tmp = list(np.unique(yte, return_counts=True)[-1] / len(yte))
    print('Prior', max(tmp))
    return d


def warmup_embedding(i, epochs, kg_model, warmup=0.1):
    # warm-up embeddings. Train only embeddings for first epochs.
    if i / epochs <= warmup:
        for l in kg_model.layers:
            if 'dense' in l.name or 'x' in l.name:
                l.trainable = False
    else:
        for l in kg_model.layers:
            l.trainable = True


def plot_loss(losses):
    losses = list(zip(*losses))
    for l in losses:
        plt.plot(l)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['KG total', 'KG score', 'KG x', 'ON x'])
    plt.show()


if __name__ == '__main__':
    main()
