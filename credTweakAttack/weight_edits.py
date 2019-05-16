import itertools
import csv, json
from multiprocessing import Pool
import time
import sys
from collections import Counter

def _counter(l):
    i, row = l
    if i % 100000 == 0:
        print("Done: {}".format(i))
    return json.loads(row[2])


CHUNK_SIZE = int(1e6)
def count_edits_parallel(fname):
    """ Slow!! """
    c = Counter()
    with open(fname) as f:
        csvf = csv.reader(f)
        with Pool() as p:
            for x in p.imap(_counter, enumerate(csvf), chunksize=CHUNK_SIZE):
                c.update(x)
    with open('transition_count.json', 'w') as f:
        json.dump(c, f, indent=4)

CHUNK_SIZE = int(1e6)
def count_edits_parallel(fname):
    c = Counter()
    with open(fname) as f:
        csvf = csv.reader(f)
        for i, row in enumerate(csvf):
            if i % 1000000 == 0:
                print("Done: {}".format(i))
            c.update(json.loads(row[2]))
    with open('transition_count.json', 'w') as f:
        json.dump(c, f, indent=4)


if __name__ == "__main__":
    count_edits_parallel(sys.argv[1])
