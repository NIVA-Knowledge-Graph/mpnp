import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy import stats
from numpy.polynomial import Polynomial
import pandas as pd
from prettytable import PrettyTable
from itertools import cycle
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from scipy import interp

def load_results():
    with open('../results/results.pkl', 'rb') as f:
        return pickle.load(f)


def add_result(name, result):
    obj = {}
    with open('../results/results.pkl', 'rb') as f:
        try:
            obj = pickle.load(f)
        except:
            pass
        obj[name] = result
    with open('../results/results.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


    # es_full_kg_distmult
    # es_full_kg_TransE

    # es_only_cid_mapped_distmult_distmult
    # es_only_cid_mapped_distmult_TransE

    # es_two_step_normalized_distmult
    # es_two_step_normalized_TransE
    # es_two_step_avg_distmult
    # es_two_step_avg_TransE

    # es_directed_one_step_back_normalized_distmult
    # es_directed_one_step_back_normalized_TransE
    # es_directed_one_step_back_avg_distmult
    # es_directed_one_step_back_avg_TransE

    # es_descending_influence_normalized_distmult
    # es_descending_influence_normalized_TransE
    # es_descending_influence_avg_distmult
    # es_descending_influence_avg_TransE

def main():
    #prefix = 'full_kg_distmult'
    #t_and_p_test(prefix)
    #plot_f1_AVG(prefix)
    #plot_AUC_poly(prefix)
    #print_stats(prefix)
    #plot_median()
    plot_AUC()


def plot_median():
    dftr = pd.read_csv('../data/LC50_train_CID.csv').dropna()
    xtr, ytr = list(dftr['cid']), list(dftr['y'])
    xtr, ytr = list(xtr), np.asarray(ytr).reshape((-1, 1))
    median = np.median(ytr)
    #median = len(ytr) // 2 #np.argsort(ytr)[len(ytr) // 2]


    #ytr = ytr - median
    pass
    ytr.sort(axis=0)
    x = range(len(ytr))
    #plt.plot(x, ytr)
    #plt.show()

    #fig = plt.figure()
    #ax = fig.add_axes([0, 0, 1, 1])
    #langs = ['C', 'C++', 'Java', 'Python', 'PHP']
    #students = [23, 17, 35, 29, 12]
    #ax.bar(x, ytr.reshape(-1,))
    #plt.show()

    fig, (ax1) = plt.subplots(1, 1, sharex=True)
    rytr = ytr.reshape(-1,)
    ax1.fill_between(x, 0, rytr, where=rytr >= median,  facecolor='green')
    ax1.fill_between(x, 0, rytr, where=rytr <= median, facecolor='red')
    plt.show()
    #ax1.set_ylabel('between y1 and 0')


def print_stats(prefix, early_stopping = False):
    tab = PrettyTable(['Name', 'F1', 'F1-d', 'P-val', 'AUC', 'AUC-d', 'Recall', 'Recall-d'])
    #t.add_row(['Alice', 24])
    #t.add_row(['Bob', 19])

    # + es_
    prefixes = ['full_kg_distmult',
    'full_kg_TransE',
    'only_cid_mapped_distmult',
    'only_cid_mapped_TransE',
    'two_step_normalized_distmult',
    'two_step_normalized_TransE',
    'two_step_avg_distmult',
    'two_step_avg_TransE',
    'directed_one_step_back_normalized_distmult',
    'directed_one_step_back_normalized_TransE',
    'directed_one_step_back_avg_distmult',
    'directed_one_step_back_avg_TransE',
    'descending_influence_normalized_distmult',
    'descending_influence_normalized_TransE',
    'descending_influence_avg_distmult',
    'descending_influence_avg_TransE']

    for prefix in prefixes:
        if early_stopping:
            prefix = 'es_' + prefix
        add_metric_row(prefix, tab)

    print(tab)


def add_metric_row(prefix, tab):
    print("adding pre: " + prefix)
    results = load_results()
    relevant_names = [k for k, v in results.items() if k.startswith(prefix)]
    relevant_results = [v for k, v in results.items() if k.startswith(prefix)]
    kg_f1_avg = np.average([v['KG x_f1'][0] for v in relevant_results])
    on_f1_avg = np.average([v['One-Hot f1'][0] for v in relevant_results])
    f1 = kg_f1_avg * 100
    f1diff = f1 - (on_f1_avg * 100)
    # print("F1:{:.1f}".format(f1))
    # print("F1 diff:{:.1f}".format(f1diff))
    kg_f1 = [v['KG x_f1'][0] for v in relevant_results]
    on_f1 = [v['One-Hot f1'][0] for v in relevant_results]
    t, p = stats.ttest_ind(np.asarray(kg_f1), np.asarray(on_f1))
    # print("p:{:.1f}".format(p*100))
    fpr_kg = np.concatenate([v['KG fpr'][0] for v in relevant_results])
    tpr_kg = np.concatenate([v['KG tpr'][0] for v in relevant_results])
    auc_kg = np.average([v['KG auc'][0] for v in relevant_results])
    fpr_on = np.concatenate([v['One-Hot fpr'][0] for v in relevant_results])
    tpr_on = np.concatenate([v['One-Hot tpr'][0] for v in relevant_results])
    auc_on = np.average([v['One-Hot auc'][0] for v in relevant_results])
    # print("AUC:{:.1f}".format(auc_kg*100))
    # print("AUC diff:{:.1f}".format((auc_kg*100)-(auc_on*100)))
    kg_recall_avg = np.average([v['KG x_recall'][0] for v in relevant_results])
    on_recall_avg = np.average([v['One-Hot recall'][0] for v in relevant_results])

    recall = kg_recall_avg * 100
    recalldiff = recall - (on_recall_avg*100)

    f1 = kg_f1_avg * 100
    f1diff = f1 - (on_f1_avg * 100)
    # print("Recall:{:.1f}".format(f1))
    # print("Recall diff:{:.1f}".format(f1diff))
    tab.add_row([prefix,
                 "{:.1f}".format(f1),
                 "{:.1f}".format(f1diff),
                 "{:.1f}".format(p * 100),
                 "{:.1f}".format(auc_kg * 100),
                 "{:.1f}".format((auc_kg * 100) - (auc_on * 100)),
                 "{:.1f}".format(recall),
                 "{:.1f}".format(recalldiff)
                 ])


def t_and_p_test(prefix):
    results = load_results()
    relevant_results = [v for k, v in results.items() if k.startswith(prefix)]
    kg_f1 = [v['KG x_f1'][0] for v in relevant_results]
    on_f1 = [v['One-Hot f1'][0] for v in relevant_results]
    t, p = stats.ttest_ind(np.asarray(kg_f1), np.asarray(on_f1))
    print("t = " + str(t))
    print("p = " + str(p))


def plot_f1_all():
    relevant_prefix = ['descending_influence_normalized_distmult',
                       'descending_influence_avg_distmult',
                       'two_step_normalized_distmult',
                       'two_step_avg_distmult',
                       'full_kg_distmult',
                       'only_cid_mapped_distmult']
    results = load_results()

    labels = ['desc',
              'desc_avg',
              '2_step',
              '2_step_avg',
              'full_kg',
              'cid_mapped']
    kg = []
    on = []
    for prefix in relevant_prefix:
        relevant_results = [v for k, v in results.items() if k.startswith(prefix)]
        kg.append(np.average([v['KG x_f1'][0] for v in relevant_results]))
        on.append(np.average([v['One-Hot f1'][0] for v in relevant_results]))

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects_kg = ax.bar(x - width / 2, kg, width, label='kg')
    rects_on = ax.bar(x + width / 2, on, width, label='on')

    ax.set_ylabel('F1 Scores')
    ax.set_title('F1 Scores by one-hot and KG models')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.1f}%'.format(height*100),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects_kg)
    autolabel(rects_on)
    fig.tight_layout()
    plt.show()


def plot_f1_AVG(prefix):
    results = load_results()
    relevant_results = [v for k, v in results.items() if k.startswith(prefix)]
    kg_f1 = np.average([v['KG x_f1'][0] for v in relevant_results])
    on_f1 = np.average([v['One-Hot f1'][0] for v in relevant_results])

    objects = ('kg_f1 (value = {:.3f})'.format(kg_f1), 'on_f1 (value = {:.3f})'.format(on_f1))
    y_pos = np.arange(len(objects))
    performance = [kg_f1, on_f1]

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Usage')
    plt.title('F1 for {} (Number of Runs = {:.0f})'.format(prefix, len(relevant_results)))

    plt.show()


def plot_all_AUC_poly():
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--')

    relevant_prefix = ['descending_influence_normalized_distmult',
                       # 'descending_influence_avg_distmult',
                       # 'two_step_normalized_distmult',
                       'two_step_avg_distmult',
                       'full_kg_distmult']
    results = load_results()
    relevant_results = [v for k, v in results.items() if k.startswith(tuple(relevant_prefix))]
    fpr_on = np.concatenate([v['One-Hot fpr'][0] for v in relevant_results])
    tpr_on = np.concatenate([v['One-Hot tpr'][0] for v in relevant_results])
    auc_on = np.average([v['One-Hot auc'][0] for v in relevant_results])
    f_on = Polynomial.fit(fpr_on, tpr_on, 10)

    full = [v for k, v in results.items() if k.startswith('full_kg_distmult')]
    fpr_full_kg = np.concatenate([v['KG fpr'][0] for v in full])
    tpr_full_kg = np.concatenate([v['KG tpr'][0] for v in full])
    f_full_kg = Polynomial.fit(fpr_full_kg, tpr_full_kg, 10)
    x_full, y_full = f_full_kg.linspace()
    for prefix in relevant_prefix:
        relevant = [v for k, v in results.items() if k.startswith(prefix)]
        fpr_kg = np.concatenate([v['KG fpr'][0] for v in relevant])
        tpr_kg = np.concatenate([v['KG tpr'][0] for v in relevant])
        auc_kg = np.average([v['KG auc'][0] for v in relevant])
        f_kg = Polynomial.fit(fpr_kg, tpr_kg, 10)
        x2, y2 = f_kg.linspace()
        plt.plot(*f_kg.linspace(), label='{} (area = {:.3f})'.format(prefix, auc_kg))
        plt.fill_between(x_full, y_full, y2, where=y2 >= y_full, interpolate=True)

    plt.plot(*f_on.linspace(), label='On (area = {:.3f})'.format(auc_on))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve All')
    plt.legend(loc='best')
    plt.show()


def plot_AUC_poly(prefix):
    results = load_results()
    relevant_results = [v for k, v in results.items() if k.startswith(prefix)]

    fpr_kg = np.concatenate([v['KG fpr'][0] for v in relevant_results])
    tpr_kg = np.concatenate([v['KG tpr'][0] for v in relevant_results])
    auc_kg = np.average([v['KG auc'][0] for v in relevant_results])

    fpr_on = np.concatenate([v['One-Hot fpr'][0] for v in relevant_results])
    tpr_on = np.concatenate([v['One-Hot tpr'][0] for v in relevant_results])
    auc_on = np.average([v['One-Hot auc'][0] for v in relevant_results])

    f_kg = Polynomial.fit(fpr_kg, tpr_kg, 10)
    f_on = Polynomial.fit(fpr_on, tpr_on, 10)

    x1, y1 = f_kg.linspace()
    x2, y2 = f_on.linspace()

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(*f_kg.linspace(), label='Kg (area = {:.3f})'.format(auc_kg))
    plt.plot(*f_on.linspace(), label='On (area = {:.3f})'.format(auc_on))
    plt.fill_between(x1, y1, y2, where=y2 >= y1, facecolor='red', interpolate=True)
    plt.fill_between(x1, y1, y2, where=y2 <= y1, facecolor='green', interpolate=True)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for {} (Number of Runs = {:.0f})'.format(prefix, len(relevant_results)))
    plt.legend(loc='best')
    plt.show()


def sortlists(l1,l2):
    l1, l2 = zip(*sorted(zip(l1, l2)))
    return l1, l2


def plot_AUC():

    results = load_results()
    lw = 2


    fpr_full, tpr_full = get_relevant_results('full_kg_distmult', results)
    fpr_full, tpr_full, auc_full = macro_avg_roc(fpr_full, tpr_full)

    fpr_two, tpr_two = get_relevant_results('two_step_avg_distmult', results)
    fpr_two, tpr_two, auc_two = macro_avg_roc(fpr_two, tpr_two)

    fpr_desc, tpr_desc = get_relevant_results('descending_influence_normalized_distmult', results)
    fpr_desc, tpr_desc, auc_desc = macro_avg_roc(fpr_desc, tpr_desc)

    fpr_cid, tpr_cid = get_relevant_results('only_cid_mapped_TransE', results)
    fpr_cid, tpr_cid, auc_cid = macro_avg_roc(fpr_cid, tpr_cid)



    #fpr_on = [v['One-Hot fpr'][0] for v in relevant_results]
    #tpr_on = [v['One-Hot tpr'][0] for v in relevant_results]
    #fpr_on2, tpr_on2, auc_on2 = macro_avg_roc(fpr_on, tpr_on, lw, 'ON')

    # Plot all ROC curves
    plt.figure(figsize=(7, 7)) #plt.figure()
    plt.plot(fpr_full["macro"], tpr_full["macro"], lw=lw, label='BLM Bin. DM')
    plt.plot(fpr_cid["macro"], tpr_cid["macro"], lw=lw, label='OCM Bin. TE')
    plt.plot(fpr_two["macro"], tpr_two["macro"], lw=lw, label='LSC Avg. DM')
    plt.plot(fpr_desc["macro"], tpr_desc["macro"], lw=lw, label='DIC Nor. DM')
    #plt.plot(fpr_cid["macro"], tpr_cid["macro"], lw=lw, label='OCM')
    #plt.plot(fpr_on2["macro"], tpr_on2["macro"], lw=lw, label='macro average {} (area = {:.3f}))'.format('ON', auc_on2))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.7, 1.0])
    plt.ylim([0.7, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()


def get_relevant_results(prefix, results):
    relevant_results = [v for k, v in results.items() if k.startswith(prefix)]
    fpr_kg = [v['KG fpr'][0] for v in relevant_results]
    tpr_kg = [v['KG tpr'][0] for v in relevant_results]
    return fpr_kg, tpr_kg


def macro_avg_roc(fpr_kg, tpr_kg):
    all_fpr = np.unique(np.concatenate(fpr_kg))
    n_classes = 7
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], = fpr_kg[i], tpr_kg[i]
        roc_auc[i] = auc(fpr[i], tpr[i])
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr_kg[i], tpr_kg[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc = auc(fpr["macro"], tpr["macro"])
    return fpr, tpr, auc(fpr["macro"], tpr["macro"])


def plot_AUC_ns(prefix):
    results = load_results()
    relevant_results = [v for k, v in results.items() if k.startswith(prefix)]

    fpr_kg = np.concatenate([v['KG fpr'][0] for v in relevant_results])
    tpr_kg = np.concatenate([v['KG tpr'][0] for v in relevant_results])
    auc_kg = np.average([v['KG auc'][0] for v in relevant_results])

    fpr_on = np.concatenate([v['One-Hot fpr'][0] for v in relevant_results])
    tpr_on = np.concatenate([v['One-Hot tpr'][0] for v in relevant_results])
    auc_on = np.average([v['One-Hot auc'][0] for v in relevant_results])

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_kg, tpr_kg, label='Kg (area = {:.3f})'.format(auc_kg))
    plt.plot(fpr_on, tpr_on, label='On (area = {:.3f})'.format(auc_on))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for {} (Number of Runs = {:.0f})'.format(prefix, len(relevant_results)))
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()
