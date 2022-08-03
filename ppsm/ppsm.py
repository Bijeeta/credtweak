#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import gensim
import numpy as np
from word2keypress import Keyboard
# from gensim.models.utils_any2vec import _save_word2vec_format, _load_word2vec_format, compute_ngrams, ft_hash_broken
from gensim.models.utils_any2vec import ft_hash_broken, compute_ngrams
from gensim import utils, matutils
from zxcvbn import zxcvbn
import psutil

thisdir = Path(__name__).absolute().parent
print(thisdir)
process = psutil.Process(os.getpid())
print("Initial memory use: {}MB".format(process.memory_info().rss/1024/1024))  # in bytes 

KB = Keyboard()

#### Big fasttext models are not used for memory concern, the following model is
#### equally accurate and small
# model_file = str(thisdir / "fastText2_keyseq_mincount:10_ngram:1-4_negsamp:5_subsamp:0.001_d:100")
# model = gensim.models.Word2Vec.load(model_file)
# model.init_sims()
model = None

#### This is a compressed model and works equally well. This is without product
#### quantization, which can further compress the model.
model_file = str(thisdir / "fastText2_keyseq_mincount:10_ngram:1-4_negsamp:5_subsamp:0.001_d:100_compressed.npz")
_t = np.load(model_file)
vectors_ngrams_norm, hash2index, (min_n, max_n) = _t['vectors_ngrams_norm'], _t['hash2index'], _t['min_max_n']

process = psutil.Process(os.getpid())
print("Final memory use: {}MB".format(process.memory_info().rss/1024/1024))  # in bytes 


def get_vector_ngram(word):
    """ Old style model that uses the large data, the small one does not need them. 
    USE the get_vector_ngram_compressed_model function
    """
    assert False, "Are you really sure you want to use this function. If so, please comment this line!"
    global model
    model_file = str(thisdir / "fastText2_keyseq_mincount:10_ngram:1-4_negsamp:5_subsamp:0.001_d:100")
    model = gensim.models.Word2Vec.load(model_file)
    model.init_sims()

    word_vec = np.zeros(model.wv.vectors_ngrams.shape[1], dtype=np.float32)
  
    ngrams = compute_ngrams(word, model.wv.min_n, model.wv.max_n)
    ngrams_found = 0
    
    for ngram in ngrams:
        ngram_hash = ft_hash_broken(ngram) % model.wv.bucket
        if ngram_hash in model.wv.hash2index:
            word_vec += model.wv.vectors_ngrams_norm[model.wv.hash2index[ngram_hash]]
            ngrams_found += 1
    if word_vec.any():
        return word_vec / max(1, ngrams_found)
    

def get_vector_ngram_compressed_model(word):
    """
    An alternative function to get_vector_ngram that uses the compressed model.
    @vector_ngrams is an np array of size n_ngrams x d_vec 
    @min_n and @max_n are constants
    """
    min_n = 1
    max_n = 4
    bucket = hash2index.shape[0]
    word_vec = np.zeros(vectors_ngrams_norm.shape[1], dtype=np.float32)
  
    ngrams = compute_ngrams(word, min_n, max_n)
    ngrams_found = 0

    for ngram in ngrams:
        ngram_hash = ft_hash_broken(ngram) % bucket
        if hash2index[ngram_hash] != -1:
            word_vec += vectors_ngrams_norm[hash2index[ngram_hash]]
            ngrams_found += 1
    if word_vec.any():
        return word_vec / max(1, ngrams_found)
    
    
def similarity(word1, word2):
    return np.dot(matutils.unitvec(get_vector_ngram_compressed_model(word1)),
               matutils.unitvec(get_vector_ngram_compressed_model(word2)))

def get_vec(word1):
    return matutils.unitvec(get_vector_ngram_compressed_model(word1))
    
def ppsm(word1, word2):
    # target word -> word2
    # input word -> word1
    zxcvbn_res = zxcvbn(word2)
    ppsm_score = zxcvbn_res['score']
    kb_word1 = KB.word_to_keyseq(word1)
    kb_word2 = KB.word_to_keyseq(word2)
    
    similarity_score = similarity(kb_word1,kb_word2)
    if similarity_score>0.5:
        ppsm_score = 0
    return ppsm_score

if __name__ == "__main__":
    input_word = sys.argv[1]
    tar_word = sys.argv[2]
    # assert get_vector_ngram_compressed_model(input_word) == get_vector_ngram(input_word)
    # assert get_vector_ngram_compressed_model(tar_word) == get_vector_ngram(tar_word)
    print(similarity(input_word,tar_word))
