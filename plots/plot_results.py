import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy import stats
from numpy.polynomial import Polynomial


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


def main():
    plot_AUC_poly('directed_one_step_back_normalized_distmult')


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


def plot_AUC(prefix):
    results = load_results()
    relevant_results = [v for k, v in results.items() if k.startswith(prefix)]

    # sorting would not be correct
    # without sorting it produces a own line for each run, making it hard to interpret
    fpr_kg = np.sort(np.concatenate([v['KG fpr'][0] for v in relevant_results]))
    tpr_kg = np.sort(np.concatenate([v['KG tpr'][0] for v in relevant_results]))
    auc_kg = np.average([v['KG auc'][0] for v in relevant_results])

    fpr_on = np.sort(np.concatenate([v['One-Hot fpr'][0] for v in relevant_results]))
    tpr_on = np.sort(np.concatenate([v['One-Hot tpr'][0] for v in relevant_results]))
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
