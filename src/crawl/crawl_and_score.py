import pandas as pd
import numpy as np
import src.crawl.directed as drc
import src.crawl.descending_influence as dic
import src.crawl.limited_step as lsc
import src.crawl.only_cid_mapped as ocm


def main():
    process_kg()


def process_kg():
    kg = get_kg()
    xtr, ytr = get_training_data()
    create_basic_scoring_algorithm_file(kg, xtr, ytr)
    create_only_cid_mapped_file(kg)
    create_limited_step_files(kg, xtr, ytr)
    create_directed_files(kg, xtr, ytr)
    create_descending_influence_files(kg, xtr, ytr)


def create_basic_scoring_algorithm_file(kg, xtr, ytr):
    __crawl_and_score(__basic_crawl, __basic_score, 'basic_scoring-algorithm', kg, xtr, ytr)


def create_only_cid_mapped_file(kg):
    ocm.create_file(kg)


def create_limited_step_files(kg, xtr, ytr):
    __crawl_and_score(lsc.one_step_crawl, __basic_score, 'one_step', kg, xtr, ytr)
    __crawl_and_score(lsc.two_step_crawl, __basic_score, 'two_step', kg, xtr, ytr)


def create_directed_files(kg, xtr, ytr):
    __crawl_and_score(drc.simple_directed_crawl, __basic_score, 'directed_simple_test', kg, xtr, ytr)
    __crawl_and_score(drc.directed_crawl_with_backstep_crawl, __basic_score, 'directed_back_step_test', kg, xtr, ytr)
    __crawl_and_score(drc.directed_crawl_with_backstep_on_first_crawl, __basic_score, 'directed_back_step_on_first_test', kg, xtr, ytr)


def create_descending_influence_files(kg, xtr, ytr):
    __crawl_and_score(__basic_crawl, dic.descending_influence_score, 'descending_influence', kg, xtr, ytr)
    __crawl_and_score(__basic_crawl, dic.descending_influence_avg_score, 'descending_influence_avg', kg, xtr, ytr)


def __crawl_and_score(crawl, score, filename, kg, xtr, ytr):
    median = np.median(ytr)
    ytr = ytr - median

    kg_dict = score(crawl, kg, xtr, ytr)

    kg_dict = sorted(kg_dict.items(), key=lambda x: x[1]['score'], reverse=True)
    kg_dict = [(sublist[0], sublist[1], sublist[2], v['score'], v['touched']) for sublist, v in kg_dict]
    kg = pd.DataFrame(kg_dict, columns=['s', 'p', 'o', 'score', 'touched'])
    kg.to_csv('kg/processed/' + filename + '.csv')


def get_training_data():
    dftr = pd.read_csv('./data/LC50_train_CID.csv').dropna()
    xtr, ytr = list(dftr['cid']), list(dftr['y'])
    xtr, ytr = list(xtr), np.asarray(ytr).reshape((-1, 1))
    return xtr, ytr


def get_kg():
    kg = pd.read_csv('kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kgmesh = pd.read_csv('kg/kg_mesh_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    kg = kg + kgmesh
    return kg


def __basic_score(crawl, kg, xtr, ytr):
    listOfv = [{'score': 0.0, 'touched': 0.0} for _ in range(len(kg))]
    kg_dict = dict(zip(kg, listOfv))
    counter = 0
    for chemical, score in zip(xtr, ytr):
        print("progress: " + str(round((counter / len(xtr) * 100), 2)) + "%")
        counter += 1
        neighbors = crawl(kg, {chemical})
        if len(neighbors[0]) > 3:
            neighbors = [(s, p, o) for s, p, o, step in neighbors]
        for neighbor in neighbors:
            kg_dict[neighbor] = {'score': kg_dict[neighbor]['score'] + score[0],
                                 'touched': kg_dict[neighbor]['touched'] + 1}
    kg_dict = {k: {'score': np.abs(v['score']), 'touched': v['touched']} for k, v in kg_dict.items()}
    return kg_dict


def __basic_crawl(kg, to_be_explored):
    global explored
    explored = set([])
    return __find_neighbors(kg, to_be_explored, 1)


def __find_neighbors(kg, to_be_explored, step):
    global explored
    if not to_be_explored:
        return []
    explored = explored | to_be_explored
    connected_by_objects = [(s, p, o, step) for s, p, o in kg if o in to_be_explored]
    s = [s for s, p, o, step in connected_by_objects]
    results_when_searching_subjects = __find_neighbors(kg, set(s) - explored, step + 1)
    connected_by_subject = [(s, p, o, step) for s, p, o in kg if s in to_be_explored]
    o = [o for s, p, o, step in connected_by_subject]
    results_when_searching_objects = __find_neighbors(kg, set(o) - explored, step + 1)
    return results_when_searching_objects + results_when_searching_subjects + connected_by_objects + connected_by_subject


if __name__ == '__main__':
    main()