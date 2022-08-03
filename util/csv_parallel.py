from multiprocessing import Pool
import itertools
import time
import json
import csv

# Set these values by calling csv_parallel.bla = ?
CHUNK_SIZE = 10000
N_CPU = 12
MAX_PW_LEN = 30
MAX_NUM_PW_PER_USER = 100


def _run_parallel(func, chunk):
    with Pool(N_CPU) as p:
        # ret = p.map(pws_list_to_list_of_paths, chunk, chunksize=CHUNK_SIZE)
        ret = p.map(func, chunk, chunksize=CHUNK_SIZE)
    return itertools.chain(*ret)


def csv2pwlist(csvf_pointer, chunksize):
    """from csvf_pointer read chunksize lines and return only the passwords as list
    of lists. This also filters out passwords that are longer than MAX_PW_LEN,
    and users that have more than MAX_NUM_PW_PER_USER passwrods

    """
    csv_slice = csv.reader(itertools.islice(csvf_pointer, 0, chunksize))
    pws_iterator = ((u, json.loads(pwlist)) for u, pwlist in csv_slice)
    # remove _long passwords
    removed_long_pws = (
        (u, [pw for pw in pws if len(pw) <= MAX_PW_LEN])
        for u, pws in pws_iterator
    )
    # removed_too_many_pws
    ret = [
        (u, pws) for u, pws in removed_long_pws if len(pws) <= MAX_NUM_PW_PER_USER
    ]
    return ret

def parallel_run_csv2dataset(csv_fpath, csv_outfile, func, start_from=0):
    """
    csv_fpath: The input csv file, should be username, json-serialied list of password format
    csv_outfile: The output file of the csv
    func: (upws) -> list-of-output; upws = (u, pws) **output put must be a list**
    start_form : from where to start csv_fpath
    """
    lines_read = start_from
    mode = 'w' if start_from <= 0 else 'a'
    with open(csv_outfile, mode) as csvfile, \
         open(csv_fpath) as csvdata:
        next(csvdata)  # skip the first line
        csvwriter = csv.writer(csvfile)
        # skip to start_from
        csvdata = itertools.islice(csvdata, start_from, None)
        while True:
            lasttime = time.time()
            chunk = csv2pwlist(csvdata, CHUNK_SIZE*N_CPU)
            if len(chunk) <= 0: break
            lines_read += len(chunk)
            print("Lines_read = {}".format(lines_read))
            csvwriter.writerows(_run_parallel(func, chunk))
            print("Done processing in {}sec".format(time.time()-lasttime))
    print("Done")
