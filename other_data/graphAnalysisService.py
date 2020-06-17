import pandas as pd
import collections


def print_graph_stat(filename):
    kg = pd.read_csv(filename, sep='\t')
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


def find_duplicates(filename):
    kg = pd.read_csv(filename, sep='\t')

    kg = list(zip(kg['s'], kg['p'], kg['o']))

    s = [s for s, p, o in kg]
    o = [o for s, p, o in kg]

    sduplicates = [item for item, count in collections.Counter(s).items() if count > 1]
    oduplicates = [item for item, count in collections.Counter(o).items() if count > 1]

    print("duplicates of subjects: " + str(len(sduplicates)))
    print("duplicates of objects: " + str(len(oduplicates)))


def main():
    print('FB15k-237-train.txt')
    print_graph_stat('FB15k-237-train.txt')
    find_duplicates('FB15k-237-train.txt')
    print('')
    print('WN18-train.txt')
    print_graph_stat('WN18-train.txt')
    find_duplicates('WN18-train.txt')


if __name__ == '__main__':
    main()
