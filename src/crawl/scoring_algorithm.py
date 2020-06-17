import pandas as pd
import numpy as np

import src.crawl.directed_crawl as drc


def main():
    create_data_structure(drc.create_data_structure())


def create_data_structure(crawl):
    kg = pd.read_csv('kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kgmesh = pd.read_csv('kg/kg_mesh_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    kg = kg + kgmesh

    dftr = pd.read_csv('./data/LC50_train_CID.csv').dropna()
    xtr, ytr = list(dftr['cid']), list(dftr['y'])
    xtr, ytr = list(xtr), np.asarray(ytr).reshape((-1, 1))
    median = np.median(ytr)
    ytr = ytr - median

    # todo: seperate the things in common so it can
    #  be used by dic drc lsc and dic-avg ect

    listOfv = [{'score': 0.0, 'touched': 0.0} for _ in range(len(kg))]
    kg_dict = dict(zip(kg, listOfv))

    counter = 0
    for chemical, score in zip(xtr, ytr):
        print("progress: " + str(round((counter/len(xtr)*100),2)) + "%")
        counter += 1
        neighbors = crawl(kg, {chemical})
        for neighbor in neighbors:
            kg_dict[neighbor] = {'score': kg_dict[neighbor]['score'] + score[0],
                                 'touched': kg_dict[neighbor]['touched'] + 1}

    kg_dict = {k: {'score': np.abs(v['score']), 'touched': v['touched']} for k, v in kg_dict.items()}
    kg_dict = sorted(kg_dict.items(), key=lambda x: x[1]['score'], reverse=True)
    kg_dict = [(sublist[0], sublist[1], sublist[2], v['score'], v['touched']) for sublist, v in kg_dict]
    kg = pd.DataFrame(kg_dict, columns=['s', 'p', 'o', 'score', 'touched'])
    kg.to_csv('kg/processed/test.csv')


if __name__ == '__main__':
    main()