import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy import stats
import pandas as pd
from prettytable import PrettyTable
from sklearn.metrics import auc
from scipy import interp


def load_results():
    with open('../results/results.pkl', 'rb') as f:
        return pickle.load(f)


def main():
    print("Results without early stopping:")
    print_stats()
    print("Results with early stopping:")
    print_stats(early_stopping=True)

    plot_the_best_AUC()


def plot_median():
    dftr = pd.read_csv('../data/LC50_train_CID.csv').dropna()
    xtr, ytr = list(dftr['cid']), list(dftr['y'])
    xtr, ytr = list(xtr), np.asarray(ytr).reshape((-1, 1))
    median = np.median(ytr)
    ytr.sort(axis=0)
    x = range(len(ytr))
    fig, (ax1) = plt.subplots(1, 1, sharex=True)
    rytr = ytr.reshape(-1,)
    ax1.fill_between(x, 0, rytr, where=rytr >= median,  facecolor='green')
    ax1.fill_between(x, 0, rytr, where=rytr <= median, facecolor='red')
    plt.show()


def print_stats(early_stopping = False):
    tab = PrettyTable(['Name', 'F1', 'F1-d', 'P-val', 'AUC', 'AUC-d', 'Recall', 'Recall-d'])
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
        __add_metric_row(prefix, tab)
    print(tab)


def __add_metric_row(prefix, tab):
    results = load_results()
    relevant_results = [v for k, v in results.items() if k.startswith(prefix)]
    kg_f1_avg = np.average([v['KG x_f1'][0] for v in relevant_results])
    on_f1_avg = np.average([v['One-Hot f1'][0] for v in relevant_results])
    kg_f1 = [v['KG x_f1'][0] for v in relevant_results]
    on_f1 = [v['One-Hot f1'][0] for v in relevant_results]
    t, p = stats.ttest_ind(np.asarray(kg_f1), np.asarray(on_f1))
    auc_kg = np.average([v['KG auc'][0] for v in relevant_results])
    auc_on = np.average([v['One-Hot auc'][0] for v in relevant_results])
    kg_recall_avg = np.average([v['KG x_recall'][0] for v in relevant_results])
    on_recall_avg = np.average([v['One-Hot recall'][0] for v in relevant_results])
    recall = kg_recall_avg * 100
    recalldiff = recall - (on_recall_avg*100)
    f1 = kg_f1_avg * 100
    f1diff = f1 - (on_f1_avg * 100)
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


def plot_the_best_AUC():
    results = load_results()
    lw = 2

    fpr_full, tpr_full = __get_relevant_results('full_kg_distmult', results)
    fpr_full, tpr_full, auc_full = __macro_avg_roc(fpr_full, tpr_full)

    fpr_two, tpr_two = __get_relevant_results('two_step_avg_distmult', results)
    fpr_two, tpr_two, auc_two = __macro_avg_roc(fpr_two, tpr_two)

    fpr_desc, tpr_desc = __get_relevant_results('descending_influence_normalized_distmult', results)
    fpr_desc, tpr_desc, auc_desc = __macro_avg_roc(fpr_desc, tpr_desc)

    fpr_cid, tpr_cid = __get_relevant_results('only_cid_mapped_TransE', results)
    fpr_cid, tpr_cid, auc_cid = __macro_avg_roc(fpr_cid, tpr_cid)

    plt.figure(figsize=(7, 7))
    plt.plot(fpr_full["macro"], tpr_full["macro"], lw=lw, label='BLM Bin. DM')
    plt.plot(fpr_cid["macro"], tpr_cid["macro"], lw=lw, label='OCM Bin. TE')
    plt.plot(fpr_two["macro"], tpr_two["macro"], lw=lw, label='LSC Avg. DM')
    plt.plot(fpr_desc["macro"], tpr_desc["macro"], lw=lw, label='DIC Nor. DM')
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()


def __get_relevant_results(prefix, results):
    relevant_results = [v for k, v in results.items() if k.startswith(prefix)]
    fpr_kg = [v['KG fpr'][0] for v in relevant_results]
    tpr_kg = [v['KG tpr'][0] for v in relevant_results]
    return fpr_kg, tpr_kg


def __macro_avg_roc(fpr_kg, tpr_kg):
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
    return fpr, tpr, auc(fpr["macro"], tpr["macro"])


if __name__ == '__main__':
    main()
