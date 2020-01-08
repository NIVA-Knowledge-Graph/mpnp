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

    results = load_results()
    for result in results:
        labels.append(result)
        kg.append(results[result]['KG x_f1'][0])
        on.append(results[result]['One-Hot f1'][0])

    x = np.arange(len(labels))
    width = 0.20

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, kg, width, label='KG')
    ax.bar(x + width / 2, on, width, label='ON')

    ax.set_ylabel('Value')
    ax.set_title(' F1 metrics')
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