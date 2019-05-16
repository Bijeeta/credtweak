
"""
Author: Tal Daniel
Minimum Edit Distance with Backtrace
-----------------------------------

We wish to find the Minimum Edit Distance (MED) between two strings. That is,
given two strings, align them, and find the minimum operations from {Insert,
Delete, Substitute} needed to get from the first string to the second string.
Then, we want to find the actual operations done in order to reach this MED,
e.g "Insert 'A' at position 3".

We can try and achieve this goal using Dynamic Programming (DP) for optimal
complexity as follows: Define:
* String 1: $X$ of length $n$
* String 2: $Y$ of length $m$
* $D[i,j]$: Edit Distance between substrings $X[1 \rightarrow i]$ and $Y[1 \rightarrow j]$

Using "Bottom Up" approach, the MED between $X$ and $Y$ would be $D[n,m]$.

We assume that the distance between string of length 0 to a string of length k
is k, since we need to insert k characters is order to create string 2.  In
order to actually find the operation, we need to keep track of the operations,
that is, create a "Backtrace".


Complexity:

* Time: O(n*m)
* Space: O(n*m)
* Backtrace: O(n+m)

"""


# Imports:

import numpy as np
import string
import json
import csv
import itertools
import time
from word2keypress import Keyboard
from ast import literal_eval
from functools import lru_cache
from pathlib import Path

thisfolder = Path(__file__).absolute().parent
TRANS_to_IDX = json.load((thisfolder / 'data/trans_dict_2idx.json').open())
IDX_to_TRANS = {v: literal_eval(k) for k, v in TRANS_to_IDX.items()}
KB = Keyboard()

CACHE_SIZE = int(1e6)


def find_med_backtrace(str1, str2, cutoff=-1):
    '''
    This function calculates the Minimum Edit Distance between 2 words using
    Dynamic Programming, and asserts the optimal transition path using backtracing.
    Input parameters: original word, target word
    Output: minimum edit distance, path
    Example: ('password', 'Passw0rd') -> 2.0, [('s', 'P', 0), ('s', '0', 5)]
    '''
    # op_arr_str = ["d", "i", "c", "s"]

    # Definitions:
    n = len(str1)
    m = len(str2)
    D = np.full((n + 1, m + 1), np.inf)
    trace = np.full((n + 1, m + 1), None)
    trace[1:, 0] = list(zip(range(n), np.zeros(n, dtype=int)))
    trace[0, 1:] = list(zip(np.zeros(m, dtype=int), range(m)))
    # Initialization:
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)

    # Fill the matrices:
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            delete = D[i - 1, j] + 1
            insert = D[i, j - 1] + 1
            if (str1[i - 1] == str2[j - 1]):
                sub = np.inf
                copy = D[i - 1, j - 1]
            else:
                sub = D[i - 1, j - 1] + 1
                copy = np.inf
            op_arr = [delete, insert, copy, sub]
            D[i, j] = np.min(op_arr)
            op = np.argmin(op_arr)
            if (op == 0):
                # delete, go down
                trace[i, j] = (i - 1, j)
            elif (op == 1):
                # insert, go left
                trace[i, j] = (i, j - 1)
            else:
                # copy or subsitute, go diag
                trace[i, j] = (i - 1, j - 1)
    # print(trace)
    # Find the path of transitions:
    i = n
    j = m
    cursor = trace[i, j]
    path = []
    while (cursor is not None):
        # 3 possible directions:
        #         print(cursor)
        if (cursor[0] == i - 1 and cursor[1] == j - 1):
            # diagonal - sub or copy
            if (str1[cursor[0]] != str2[cursor[1]]):
                # substitute
                path.append(("s", str2[cursor[1]], cursor[0]))
            i = i - 1
            j = j - 1
        elif (cursor[0] == i and cursor[1] == j - 1):
            # go left - insert
            path.append(("i", str2[cursor[1]], cursor[0]))
            j = j - 1
        else:
            # (cursor[0] == i - 1 and cursor[1] == j )
            # go down - delete
            path.append(("d", None, cursor[0]))
            i = i - 1
        cursor = trace[cursor[0], cursor[1]]
        # print(len(path), cursor)
    md = D[n, m]
    del D, trace
    return md, list(reversed(path))


# Minimum Edit Distance with Backtrace
@lru_cache(maxsize=CACHE_SIZE)
def find_med_backtrace_kb(str1, str2):
    '''
    This function calculates the Minimum Edit Distance between 2 words using
    Dynamic Programming, and asserts the optimal transition path using backtracing.
    This version uses KeyPress representation.
    Input parameters: original word, target word
    Output: minimum edit distance, path
    Example:
    ('password', 'PASSword') -> 2.0 , [('i', '\x04', 0), ('i', '\x04', 4)]
    '''
    # Transform to keyboard representation:
    kb_str1 = KB.word_to_keyseq(str1)
    kb_str2 = KB.word_to_keyseq(str2)
    return find_med_backtrace(kb_str1, kb_str2)


# Decoder - given a word and a path of transition, recover the final word:
def path2word(word, path):
    '''This function decodes the word in which the given path transitions the input
    word into.  Input parameters: original word, transition path Output: decoded
    word

    '''
    if not path:
        return word
    final_word = []
    word_len = len(word)
    path_len = len(path)
    i = 0
    j = 0
    while (i < word_len or j < path_len):
        if (j < path_len and path[j][2] == i):
            if (path[j][0] == "s"):
                # substitute
                final_word.append(path[j][1])
                i += 1
                j += 1
            elif (path[j][0] == "d"):
                # delete
                i += 1
                j += 1
            else:
                # "i", insert
                final_word.append(path[j][1])
                j += 1
        else:
            final_word.append(word[i])
            i += 1
    return ''.join(final_word)


# Decoder - given a word and a path of transition, recover the final word:
# KEYPRESS Version
def path2word_kb(word, path):
    '''
    This function decodes the word in which the given path transitions the input word into.
    This is the KeyPress version, which handles the keyboard representations.
    Input parameters: original word, transition path
    Output: decoded word
    '''
    word = KB.word_to_keyseq(word)
    if not path:
        return KB.keyseq_to_word(word)
    final_word = []
    word_len = len(word)
    path_len = len(path)
    i = 0
    j = 0
    while (i < word_len or j < path_len):
        if (j < path_len and path[j][2] == i):
            if (path[j][0] == "s"):
                # substitute
                final_word.append(path[j][1])
                i += 1
                j += 1
            elif (path[j][0] == "d"):
                # delete
                i += 1
                j += 1
            else:
                # "i", insert
                final_word.append(path[j][1])
                j += 1
        else:
            final_word.append(word[i])
            i += 1
    return (KB.keyseq_to_word(''.join(final_word)))


def generate_transition_dict():
    '''Generate a dictionary of all possible paths in a JSON format
    Assumptions: words' max length is 30 chars and words are comprised of 98 
    available characters
    'd' - ('d', None, 0-30) -> 31 options
    's' - ('s', 0-95, 0-30) -> 98x31 = 3038 options
    'i' - ('i', 0-95, 0-30) -> 98x31 = 3038 options
    Size of table: 31 + 3038 + 3038 = 6107

    # Note, because we are using keyboard sequence we the length of keypress
    # sequence can be twice the size of the password,
    Therefore, 30*2 + 1 is the max_len
    '''
    max_len = 61
    d_list = [('d', None, i) for i in range(max_len)]
    asci = list(string.ascii_letters)
    punc = list(string.punctuation)
    dig = list(string.digits)
    chars = asci + punc + dig + [" ", "\t", "\x03", "\x04"]
    s_list = [('s', c, i) for c in chars for i in range(max_len)]
    i_list = [('i', c, i) for c in chars for i in range(max_len)]

    transition_table = d_list + s_list + i_list
    transition_dict_2idx = {}
    transition_dict_2path = {}
    for i in range(len(transition_table)):
        transition_dict_2idx[str(transition_table[i])] = i
        transition_dict_2path[i] = str(transition_table[i])
    with open('data/trans_dict_2idx.json', 'w') as outfile:
        json.dump(transition_dict_2idx, outfile)
    with open('data/trans_dict_2path.json', 'w') as outfile:
        json.dump(transition_dict_2path, outfile)
    print("Transitions dictionary created as trans_dict_2idx.json & "
          "trans_dict_2path.json")
    '''
    Read:
    if filename:
        with open(filename, 'r') as f:
            transition_dict = json.load(f)
    '''


def path2idx(path):
    '''
    This functions converts human-readable transition path to a
    dictionary-indices path (for future use in RNNs).
    Input parameters: human-readable path, dictionary
    Output: dictionary-indices path
    [('i', '\x04', 0), ('s', '\x03', 1), ('i', '2', 2), ('i', '\x04', 4)] ->
    [6076, 3008, 5737, 6080]
    '''
    idx_path = [TRANS_to_IDX.get(str(p), -1) for p in path]
    return idx_path


def idx2path(path):
    '''
    This functions converts dictionary-indices transition path to a
    human-readable path (for future use in RNNs).
    Input parameters: human-readable path, dictionary
    Output: dictionary-indices path
    [6076, 3008, 5737, 6080] ->
    [('i', '\x04', 0), ('s', '\x03', 1), ('i', '2', 2), ('i', '\x04', 4)]
    '''
    str_path = [IDX_to_TRANS.get(str(p), ("<unk>", "<unk>", -1))
                for p in path]
    return str_path


def csv2pws_pairs_gen(csv_fpath, line_s=0, line_e=None):
    '''Generator function to parse the csv file, such that every row is a list of
    username and a string of passwords list.  Using itertools, find all the
    combinations of passwords, and generate an appropriate path.  For every
    password and path, build the output password, and compare the result with
    the original pair.  

    Input parameter
    @csv_fpath: path to original dataset csv
    line_s: starting line of the file
    line_e: ending line of the file
    '''
    #     csv_fpath = './sample_username_list_tr.csv'
    #     pws_pairs = []
    with open(csv_fpath) as csv_file:
        # skip lines csv_file
        csv_reader = csv.reader(
            itertools.islice(csv_file, line_s, line_e), delimiter=','
        )
        for i, row in enumerate(csv_reader):
            if (len(row) != 2):
                print("File format error @ line {}\n{!r}!".format(i, row))
                break
            username, pws_string = row
            # pws_list = eval(pws_string) # In case arrays are not json
            # formatted
            try:
                pws_list = json.loads(pws_string)
            except json.decoder.JSONDecodeError as ex:
                print(ex)
                continue
            for p in itertools.permutations(pws_list, 2):
                yield p


# def csv2dataset_gen(csv_fpath):
#     '''This (generator) function generates the new dataset format from the original
#     one.  The new dataset is in the form: [pass1, pass2, human-readable
#     transition path].  Input parameter: path to original dataset csv

#     '''
#     print("Started building dataset...")
#     start = time.clock()
#     pairs_generator = csv2pws_pairs_gen(csv_fpath)
#     with open('trans_dataset.csv', 'w', newline='') as csvfile:
#         csv_writer = csv.writer(
#             csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL
#         )
#         for i, pair in enumerate(pairs_generator):
#             med, path = find_med_backtrace_kb(pair[0], pair[1])
#             # Run test on randomly sampled 1% cases
#             if random.randint(0, 1000) <= 10:
#                 decoded_word = path2word_kb(pair[0], path)
#                 str_path = [str(p) for p in path]
#                 if (decoded_word != pair[1]):
#                     print("Test failed on: {}".format(pair))
#                     print("Path chosen: {}".format(path))
#                     print("Decoded Password: {}".format(decoded_word))
#             if (i % 50000 == 0):
#                 print("Progress: processed {} pairs so far".format(i))
#             csv_writer.writerow([pair[0], pair[1], json.dumps(str_path)])
#     print("Dataset created in {} seconds on a total of {} passwords pairs".format(
#         time.clock() -
#         start,
#         i
#     ))
#     print("New Dataset CSV file: trans_dataset.csv")

def csv2dataset_dict_gen(csv_fpath, human_readable=False):
    '''This (generator) function generates the new dataset format from the original
    one.

    The new dataset is in the form: [pass1, pass2, dictionary-indices transition
    path].

    Input parameter: path to original dataset csv, path to
    the json dictionary file

    '''
    print("Started building dataset...")
    start = time.clock()
    pairs_generator = csv2pws_pairs_gen(csv_fpath)
    with open('trans_dataset.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(
            csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for i, pair in enumerate(pairs_generator):
            if not i:
                # skip first line
                continue
            if (len(pair[0]) > 30 or len(pair[1]) > 30):
                continue
            path_indices = pair2path(pair, human_readable)
            if (i % 50000 == 0):
                print("Progress: processed {} pairs so far".format(i))
            csv_writer.writerow([
                pair[0], pair[1], json.dumps(path_indices)
            ])
    print(
        "Dataset created in {} seconds on a total of {} passwords pairs".format(
            time.clock() -
            start,
            i))
    print("New Dataset CSV file: trans_dataset.csv")


def csv2trans_dataset_gen(dataset_csv, dict_json):
    '''This (generator) function reads and processes the new dataset from a csv
    file, and parses the human-readable transition path into a dictionary
    indices transition path, using an input dictionary file.  A sample is now a
    tuple in the form of: [pass1, pass2, human-readable transition path,
    dictionary indices transition path] The csv file is in the form: [pass1,
    pass2, human-readable transition path] Input parameters: path to the dataset
    csv file, path to the json dictionary file.  Output: yields a tuple of the
    mentiond form

    '''
    with open(dict_json, 'r') as f:
        trans_dict = json.load(f)
    with open(dataset_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if (len(row) != 3):
                print("ERROR: File format error!")
                print(row)
                continue
            pass_1, pass_2, pws_str = row
            pws_list = json.loads(pws_str)
            pws_indices = path2idx(pws_list, trans_dict)
            yield (pass_1, pass_2, pws_list, pws_indices)


def csv2trans_dataset_dict_gen(dataset_csv, dict2path_json):
    '''This (generator) function reads and processes the new dataset from a csv
    file, and parses the dictionary indices transition path into a
    human-readable path.  using an input dictionary file.  A sample is now a
    tuple in the form of: [pass1, pass2, human-readable transition path,
    dictionary indices transition path] The csv file is in the form: [pass1,
    pass2, human-readable transition path] Input parameters: path to the dataset
    csv file, path to the json dictionary file.  Output: yields a tuple of the
    mentiond form

    '''
    with open(dataset_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if (len(row) != 3):
                print("File format error!")
                break
            pass_1, pass_2, pws_indices = row
            pws_list = idx2path(pws_indices)
            yield (pass_1, pass_2, pws_list, pws_indices)


def run_test(csv_fpath):
    '''
    This function tests the encoder-decoder functions, in order to make sure
    that for evey transition path from pass1 to pass2, the decoded password from pass1
    and the transition path is the same as pass2.
    '''
    start = time.clock()
    pws_pairs_gen = csv2pws_pairs_gen(csv_fpath)
    for i, pair in enumerate(pws_pairs_gen):
        med, path = find_med_backtrace_kb(pair[0], pair[1])
        decoded_word = path2word_kb(pair[0], path)
        if (decoded_word != pair[1]):
            print("Test failed on: {}".format(pair))
            print("Path chosen: {}".format(path))
            print("Decoded Password: {}".format(decoded_word))
        if i % 100 == 0:
            print("Done: {}".format(i))
    print("Testing done in {} seconds on a total of {} passwords pairs"
          .format(time.clock() - start, i))




'''
Steps: When running on the server

1. Generate dictionaries using: generate_transition_dict() [creates 2 dictionaries: path2idx, idx2path]
2. Create the new dataset using: csv2dataset_dict_gen(csv_fpath, dict2idx_json)
3. In order to generate samples: samples_generator = csv2trans_dataset_dict_gen(dataset_csv, dict2path_json)
4. Samples are in the form: [pass1, pass2, human-readable transition path, dictionary indices transition path],
    take what cells you need for training/testing
'''

if __name__ == "__main__":
    import sys
    run_test(sys.argv[1])
