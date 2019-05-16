import sys
from os import path
from edit_distance_backtrace import csv2dataset_dict_gen
from edit_distance_backtrace import (
    find_med_backtrace, path2word, path2idx, KB
)
import itertools
import random
import csv
import json
from multiprocessing import Pool
import time
import functools
from pathlib import Path

homedir = Path(__file__).resolve().parent.parent
sys.path.append(str(homedir))

def main():
    if (len(sys.argv) != 3):
        print("Arguments: 1. Path to csv file ('./fname.csv') 2. Path to directory of dictionary path->idx ('./dict/')")
        raise SystemExit(
            "No path to csv file and/or dictionaries supplied. Exiting...")
    for arg in sys.argv:
        print(arg)
    path_to_csv = sys.argv[1]
    path_to_dict = sys.argv[2]

    csv2dataset_dict_gen(
        path_to_csv, path.join(
            path_to_dict, 'trans_dict_2idx.json'))


################################################################################
#### Parallel generation #######################################################
################################################################################
CHUNK_SIZE = 10000
N_CPU = 12
MAX_PW_LEN = 30
MAX_NUM_PW_PER_USER = 100


def pair2path(kw1, kw2, human_readable=False):
    """given a pair of passwords (kw1, kw2) returns one of the possible list of 
    transformations (paths) that converts pair[0] into  other[1].
    kw1 and kw2 are already in keyboard sequence format.

    Returns a list of transoformation indices.  If human_readable is true, then
    returns the list of raw transformations.

    E.g,
    >> pairs2path('\x02password', 'password')
    ['d', 0, None]
    """
    med, path = find_med_backtrace(kw1, kw2)
    if human_readable:
        path_indices = [str(p) for p in path]
    else:
        path_indices = path2idx(path)
    # for testing
    if random.randint(0, 1000) <= 10:
        decoded_word = path2word(kw1, path)
        if (decoded_word != kw2):
            print("Test failed on: {}".format((kw1, kw2)))
            print("Path chosen: {}".format(path))
            print("Decoded Password: {}".format(decoded_word))

    return path_indices


# def pws_list_to_list_of_paths(pwlist):
#     """given a list of list of pws, return all pair paths"""
#     print("Len(pwlist) = {}".format(len(pwlist)))
#     print(pwlist[:10])
#     d = [
#         (w1, w2, pair2path(kw1, kw2))
#         for pwl in pwlist
#         for ((w1, kw1), (w2, kw2)) in itertools.permutations(pwl, 2)
#     ]
#     print("d = {}".format(len(d)))
#     return d


# def csv2pwlist(csvf_pointer, chunksize):
#     """from csvf_pointer read chunksize lines and return only the passwords as list
#     of lists. This also filters out passwords that are longer than MAX_PW_LEN,
#     and users that have more than MAX_NUM_PW_PER_USER passwrods

#     """
#     csv_slice = csv.reader(itertools.islice(csvf_pointer, 0, chunksize))
#     pws_iterator = (json.loads(pwlist) for u, pwlist in csv_slice)
#     # remove _long passwords
#     removed_long_pws = (
#         [(pw, KB.word_to_keyseq(pw)) for pw in pws if len(pw) <= MAX_PW_LEN]
#         for pws in pws_iterator
#     )
#     # removed_too_many_pws
#     ret = [
#         pws for pws in removed_long_pws if len(pws) <= MAX_NUM_PW_PER_USER
#     ]
#     return ret

def _pws_to_path(upws):
    u, pws = upws
    kws = {w1: KB.word_to_keyseq(w1) for w1 in set(pws)}
    return [
        (w1, w2, json.dumps(pair2path(kws[w1], kws[w2])))
        for (w1, w2) in itertools.permutations(pws, 2)
    ]

# def _run_parallel(chunk):
#     with Pool(N_CPU) as p:
#         # ret = p.map(pws_list_to_list_of_paths, chunk, chunksize=CHUNK_SIZE)
#         ret = p.map(_pws_to_path, chunk, chunksize=CHUNK_SIZE)
#     return itertools.chain(*ret)

# def parallel_run_csv2dataset(csv_fpath, csv_outfile, start_from=0):
#     lines_read = start_from
#     mode = 'w' if start_from <= 0 else 'a'
#     with open(csv_outfile, mode) as csvfile, \
#          open(csv_fpath) as csvdata:
#         next(csvdata)  # skip the first line
#         csvwriter = csv.writer(csvfile)
#         # skip to start_from
#         csvdata = itertools.islice(csvdata, start_from, None)
#         while True:
#             lasttime = time.time()
#             chunk = csv2pwlist(csvdata, CHUNK_SIZE*N_CPU)
#             if len(chunk) <= 0: break
#             lines_read += len(chunk)
#             print("Lines_read = {}".format(lines_read))
#             csvwriter.writerows(_run_parallel(chunk))
#             print("Done processing in {}sec".format(time.time()-lasttime))
#     print("Done")


@functools.lru_cache(maxsize=100000)
def _pass2kb(w):
    try:
        return KB.word_to_keyseq(w)
    except (KeyError, ValueError) as ex:
        print("ERROR: {!r} ---> {}".format(w, ex))


def typo_pairs_to_path(fname):
    import gzip
    outfname = fname.replace('.csv.gz', 'paths.csv')
    with gzip.open(fname, 'rt') as f, open(outfname, 'w') as of:
        csvf = csv.reader((l.replace('\0', '') for l in f))
        outcsv = csv.writer(of)
        outcsv.writerow(['tpw', 'rpw', 'path', 'source'])
        outcsv.writerows(
            (w2, w1, json.dumps(pair2path(_pass2kb(w2), _pass2kb(w1))))
            for w1, w2, source in csvf
        )

        

if __name__ == "__main__":
    # print(find_med_backtrace_kb('PASSWORD', 'Password'))
    # run_test(sys.argv[1])
    
    ## For running the csv to paths in parallel

    csv_data_file = sys.argv[1]
    csv_outfile = sys.argv[2]
    start_from = 0 if len(sys.argv) < 4 else int(sys.argv[3])
    ## TODO : Test this change
    import util.csv_parallel as csvp
    csvp.parallel_run_csv2dataset(
        csv_data_file, csv_outfile, _pws_to_path, start_from
    )

    ## For typo data
    typo_pairs_to_path(sys.argv[1])
    # main()
