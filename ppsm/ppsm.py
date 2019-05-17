import gensim
from gensim.models import FastText
import numpy as np
from word2keypress import Keyboard
from numpy import dot
from gensim.models.utils_any2vec import _save_word2vec_format, _load_word2vec_format, _compute_ngrams, _ft_hash
import math
import sys
from gensim import utils, matutils
from zxcvbn import zxcvbn
  
KB = Keyboard()
model_file = sys.argv[1]
model = gensim.models.Word2Vec.load(model_file)
model.init_sims()

def get_vector_ngram(word):
    word_vec = np.zeros(model.wv.vectors_ngrams.shape[1], dtype=np.float32)
  
    ngrams = _compute_ngrams(word, model.wv.min_n, model.wv.max_n)
    ngrams_found = 0
    
    for ngram in ngrams:
        ngram_hash = _ft_hash(ngram) % model.wv.bucket
        if ngram_hash in model.wv.hash2index:
            word_vec += model.wv.vectors_ngrams_norm[model.wv.hash2index[ngram_hash]]
            ngrams_found += 1
    if word_vec.any():
        return word_vec / max(1, ngrams_found)
    
def similarity(word1,word2):
    return dot(matutils.unitvec(get_vector_ngram(word1)), matutils.unitvec(get_vector_ngram(word2)))

def get_vec(word1):
    return matutils.unitvec(get_vector_ngram(word1))
    
def ppsm(word1,word2):
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
input_word = sys.argv[2]
tar_word = sys.argv[3]
print(ppsm(input_word,tar_word))
    
    