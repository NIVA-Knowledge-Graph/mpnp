import pandas as pd
import collections


def print_graph_stat():
    kg = pd.read_csv('FB15k-237-train.txt', sep='\t')
    kg = list(zip(kg['s'], kg['p'], kg['o']))

    s = [s for s, p, o in kg]
    o = [o for s, p, o in kg]

    subjects = set([s for s, p, o in kg])
    objects = set([o for s, p, o in kg])
    entities = set([s for s, p, o in kg]) | set([o for s, p, o in kg])

    leaf = objects - subjects
    roots = subjects - objects
    others = objects & subjects

    print("Number of leaves: " + str(len(leaf))
          + ", roots: " + str(len(roots))
          + ", others: " + str(len(others))
          + ". In total: " + str(len(entities)))

    print("len sub:" + str(len(subjects)) + " len object: " + str(len(objects)))


def find_duplicates():
    kg = pd.read_csv('FB15k-237-train.txt', sep='\t')

    kg = list(zip(kg['s'], kg['p'], kg['o']))
    #kg = [(s, p, o) for s, p, o, in kg if p != 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']

    s = [s for s, p, o in kg]
    o = [o for s, p, o in kg]

    sduplicates = [item for item, count in collections.Counter(s).items() if count > 1]
    oduplicates = [item for item, count in collections.Counter(o).items() if count > 1]

    print("dublicates of subjects: " + str(len(sduplicates)))
    print("dublicates of obejcts: " + str(len(oduplicates)))

def main():
    print_graph_stat()
    find_duplicates()


if __name__ == '__main__':
    main()