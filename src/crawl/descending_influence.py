import numpy as np


def descending_influence_score(crawl, kg, xtr, ytr):
    listOfv = [{'score': 0.0, 'touched': 0.0} for _ in range(len(kg))]
    kg_dict = dict(zip(kg, listOfv))
    counter = 0
    for chemical, score in zip(xtr, ytr):
        print("progress: " + str(round((counter / len(xtr) * 100), 2)) + "%")
        counter += 1
        neighbors = crawl(kg, {chemical})
        triples = [(s, p, o) for s, p, o, step in neighbors]
        steps = [step for s, p, o, step in neighbors]
        for triple, step in zip(triples, steps):
            kg_dict[triple] = {'score': kg_dict[triple]['score'] + score[0] / step,
                               'touched': kg_dict[triple]['touched'] + 1}
    kg_dict = {k: {'score': np.abs(v['score']), 'touched': v['touched']} for k, v in kg_dict.items()}
    return kg_dict


def descending_influence_avg_score(crawl, kg, xtr, ytr):
    listOfv = [{'score': [], 'weight': [], 'touched': 0.0}] * len(kg)
    kg_dict = dict(zip(kg, listOfv))
    counter = 0
    for chemical, score in zip(xtr, ytr):
        print("progress: " + str(round((counter / len(xtr) * 100), 2)) + "%")
        counter += 1
        neighbors = crawl(kg, {chemical})
        triples = [(s, p, o) for s, p, o, step in neighbors]
        steps = [step for s, p, o, step in neighbors]
        for triple, step in zip(triples, steps):
            kg_dict[triple] = {'score': kg_dict[triple]['score'] + [score[0]],
                               'weight': kg_dict[triple]['weight'] + [1 / step if step else 0],
                               'touched': kg_dict[triple]['touched'] + 1}
    kg_dict = {k:
        {
            'score': np.abs(np.average(v['score'], weights=v['weight'])) if v['score'] else 0,
            'touched': v['touched']
        }
        for k, v in kg_dict.items()}
    return kg_dict
