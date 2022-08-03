"""
Modify the way fastext store and restore the vectors.
Or, 
extract some more information from the ngrams for further studies.
"""

import numpy as np
import gensim
from gensim.models.utils_any2vec import _save_word2vec_format, _load_word2vec_format, _compute_ngrams, _ft_hash
import marisa_trie
import itertools
from collections import defaultdict, Counter

def ftext_extract_ngrams(model, compressed=True):
    """
    Create a file containing all ngrams and their vectors
    """
    ngram_freq = defaultdict(int)
    for w, v in model.wv.vocab.items():
        for ng in _compute_ngrams(w, model.wv.min_n, model.wv.max_n):
            ngram_freq[ng] += v.count
    for ng, f in ngram_freq.items():
        pass
    if compressed:
        np.savez_compressed(file='model_ngrams.npz', vectors=model.wv.vectors_ngrams, ngrams=np.array(list(ngram_freq.keys())))
    else:
        ng_indexes = ['' for _ in range(len(model.wv.vectors_ngrams))]
        for ng, v in ngram_freq.items():
            ng_hash = model.wv.hash2index.get(_ft_hash(ng) % model.wv.bucket)
            if ng_hash and \
               (ng_indexes[ng_hash] == '' or ngram_freq[ng_indexes[ng_hash]] < ngram_freq[ng]):
                    ng_indexes[ng_hash] = ng
        d = pd.DataFrame(model.wv.vectors_ngrams, index=ng_indexes)
        d.to_csv('model_ngrams.csv.gz', compression='gzip', index_label="ngram")
