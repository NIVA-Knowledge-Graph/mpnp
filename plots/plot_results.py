import matplotlib.pyplot as plt
import pickle
import numpy as np


def load_results():
    with open('../results/results.pkl', 'rb') as f:
        return pickle.load(f)


def plot_results():
    labels = []

    kg = []
    on = []

    metrictype = "loss"
    results = load_results()

    results2 = ['full_kg_v1', 'full_kg_v2', 'full_kg_v3', 'full_kg_v4']#'full_kg_2th', 'full_kg_3th', 'full_kg_4th'] #'reduced_kg_V3_g6', 'reduced_kg_V3_g6_2th', 'reduced_kg_V3_g6_3th', 'neighbors', 'neighbors_v2', 'neighbors_v3',
                #]
                #'new_neighbors_v1', 'new_neighbors_v2', 'new_neighbors_v3', 'new_neighbors_v4'] #'reduced_kg', 'reduced_kg_2th', 'reduced_kg_3th', 'reduced_kg_150',
    for result in results2:
        #labels.append(result + " f1")
        #kg.append(results[result]['KG x_f1'][0])
        #on.append(results[result]['One-Hot f1'][0])

        labels.append(result + " " + metrictype)
        kg.append(results[result]['KG x_' + metrictype][0])
        on.append(results[result]['One-Hot ' + metrictype][0])

    x = np.arange(len(labels))
    width = 0.20

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, kg, width, label='KG')
    ax.bar(x + width / 2, on, width, label='ON')

    ax.set_ylabel('Value')
    ax.set_title(metrictype + ' metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()

    """
    results = load_results()

    lables = ['reduced', 'full']
    performance = []
    objects = []
    metrics = ['One-Hot f1', 'KG x_f1']

    #for k in d:
    #    print(k, d[k])

    for result in results:
        for metric in metrics:
            objects.append(result + ': ' + metric)
            a = results[result][metric][0]
            performance.append(results[result][metric][0])

    y_pos = np.arange(len(objects))
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos,objects)
    plt.ylabel('metric')
    plt.title('Results')
    plt.show()
    """


def main():
    plot_results()


if __name__ == '__main__':
    main()
