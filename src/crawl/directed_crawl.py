import pandas as pd
import numpy as np


def main():
    create_data_structure()


def create_data_structure():
    kg = pd.read_csv('./kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kgmesh = pd.read_csv('./kg/kg_mesh_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    kg = kg + kgmesh

    listOfv = [{'score': 0.0, 'touched': 0.0} for _ in range(len(kg))]
    kg_dict = dict(zip(kg, listOfv))

    dftr = pd.read_csv('./data/LC50_train_CID.csv').dropna()
    xtr, ytr = list(dftr['cid']), list(dftr['y'])
    xtr, ytr = list(xtr), np.asarray(ytr).reshape((-1, 1))
    median = np.median(ytr)

    ytr = ytr - median
    counter = 0
    for chemical, score in zip(xtr, ytr):
        print("progress: " + str(round((counter/len(xtr)*100),2)) + "%")
        counter += 1
        neighbors = find_neighbors_and_reset(kg, {chemical})
        pass
        for neighbor in neighbors:
            kg_dict[neighbor] = {'score': kg_dict[neighbor]['score'] + score[0],
                                 'touched': kg_dict[neighbor]['touched'] + 1}

    kg_dict = {k: {'score': np.abs(v['score']), 'touched': v['touched']} for k, v in kg_dict.items()}
    kg_dict = sorted(kg_dict.items(), key=lambda x: x[1]['score'], reverse=True)
    kg_dict = [(sublist[0], sublist[1], sublist[2], v['score'], v['touched']) for sublist, v in kg_dict]
    kg = pd.DataFrame(kg_dict, columns=['s', 'p', 'o', 'score', 'touched'])
    kg.to_csv('processed/directed_back_step_on_first.csv')


explored = set([])


def find_neighbors_and_reset(kg, to_be_explored):
    pass
    global explored
    explored = set([])
    directed = find_neighbors_directed(kg, to_be_explored)
    explored = set([])
    around = find_neighbors(kg, to_be_explored)
    return set(directed + around)


def find_neighbors_directed(kg, to_be_explored):
    global explored
    if not to_be_explored:
        return []
    explored = explored | to_be_explored
    connected_by_objects = [(s, p, o) for s, p, o in kg if o in to_be_explored]
    s = [s for s, p, o in connected_by_objects]
    results_when_searching_subjects = find_neighbors_one_way_recursive(kg, set(s) - explored, 'subject')
    connected_by_subject = [(s, p, o) for s, p, o in kg if s in to_be_explored]
    o = [o for s, p, o in connected_by_subject]
    results_when_searching_objects = find_neighbors_one_way_recursive(kg, set(o) - explored, 'object')
    return results_when_searching_objects + results_when_searching_subjects + connected_by_objects + connected_by_subject


def find_neighbors(kg, to_be_explored):
    global explored
    if not to_be_explored:
        return []
    explored = explored | to_be_explored
    connected_by_objects = [(s, p, o) for s, p, o in kg if o in to_be_explored]
    s = [s for s, p, o in connected_by_objects]
    results_when_searching_subjects = find_neighbors_non_recursive(kg, set(s) - explored)
    connected_by_subject = [(s, p, o) for s, p, o in kg if s in to_be_explored]
    o = [o for s, p, o in connected_by_subject]
    results_when_searching_objects = find_neighbors_non_recursive(kg, set(o) - explored)
    return results_when_searching_objects + results_when_searching_subjects + connected_by_objects + connected_by_subject


def find_neighbors_non_recursive(kg, to_be_explored):
    global explored
    if not to_be_explored:
        return []
    explored = explored | to_be_explored
    connected_by_objects = [(s, p, o) for s, p, o in kg if o in to_be_explored]

    connected_by_subject = [(s, p, o) for s, p, o in kg if s in to_be_explored]

    return connected_by_objects + connected_by_subject


def find_neighbors_one_way_recursive(kg, to_be_explored, direction='none'):
    global explored
    if not to_be_explored:
        return []
    explored = explored | to_be_explored
    if direction == 'subject':
        connected_by_objects = [(s, p, o) for s, p, o in kg if o in to_be_explored]
        s = [s for s, p, o in connected_by_objects]
        connected_by_direction = find_neighbors_one_way_recursive(kg, set(s) - explored, 'subject')
        return connected_by_objects + connected_by_direction
    if direction == 'object':
        connected_by_subject = [(s, p, o) for s, p, o in kg if s in to_be_explored]
        o = [o for s, p, o in connected_by_subject]
        connected_by_direction = find_neighbors_one_way_recursive(kg, set(o) - explored, 'object')
        return connected_by_subject + connected_by_direction


def find_neighbors_one_way_recursive_back_step(kg, to_be_explored, direction='none'):
    global explored
    if not to_be_explored:
        return []
    explored = explored | to_be_explored
    connected_by_objects = [(s, p, o) for s, p, o in kg if o in to_be_explored]
    connected_by_subject = [(s, p, o) for s, p, o in kg if s in to_be_explored]
    connected_by_direction = []
    if direction == 'subject':
        s = [s for s, p, o in connected_by_objects]
        connected_by_direction = find_neighbors_one_way_recursive(kg, set(s) - explored, 'subject')
    if direction == 'object':
        o = [o for s, p, o in connected_by_subject]
        connected_by_direction = find_neighbors_one_way_recursive(kg, set(o) - explored, 'object')
    return connected_by_objects + connected_by_subject + connected_by_direction


if __name__ == '__main__':
    main()
