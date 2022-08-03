import pandas as pd
import numpy as np

GUESS_RANK_ERR = -1
def guess_rank(model, w, targets, topn):
    """
    Finds rank of the passwords in the @targets. 
    @w is a given password
    @model is a model with the following two functions, 
    similarity, and most_similar
    """
    try:
        f = model.most_similar(w, topn=topn)
        targets_sim = [model.similarity(w, target) for target in targets]
        ranks = topn+1 - np.searchsorted([p for w,p in f][::-1], targets_sim, side='left')
        return ranks
    except KeyError as ex:
        print(ex)
        return [GUESS_RANK_ERR] * topn
    except ValueError as ex:
        print(ex, model, w, targets, topn)
        raise(ex)

def guess_rank_one(model, w, target, topn):
    return guess_rank(model, w, [target], topn)[0]



def guess_rank_fast(model, w, targets, topn):
    """
    Finds rank of the passwords in the @targets. 
    @w is a given password
    @model is a model with the following two functions, 
    similarity, and most_similar
    """
    if model.wv.vector_vocab_norm is None:
        model.inti_sims()
    try:
        word_v = model.get_vector(w, use_norm=True)
        f = np.sort(-np.dot(model.wv.vector_vocab_norm, word_v))
        targets_sim = [model.similarity(w, target) for target in targets]
        ranks = f.shape[0] - np.searchsorted(f, targets_sim, side='left')
        return ranks
    except KeyError as ex:
        print(ex)
        return [GUESS_RANK_ERR] * len(targets)
    except ValueError as ex:
        print(ex, model, w, targets, topn)
        raise(ex)


def most_similar(mf, x, topn):
    try:
        # return mf.wv.most_similar(x, topn=topn, indexer=indexer)
        ##  Indexer gives approximate results, and strictly not good.
        return mf.wv.most_similar(x, topn=topn, indexer=None)
    except KeyError as ex:
        print(ex)
        return []
    except ValueError as ex:
        print(x)
        raise(ex)


def isthere(pws, glist):
    """
    if pw in guesses 
    @pws: array of pw
    @glist: array of guesses (which is array of passwords)
    returns: array of boolean
    """
    return np.array([(pw in g) for (pw, g) in zip(pws, glist)])


from sklearn.model_selection import KFold, ParameterGrid
from multiprocessing import Pool

def optimize(func, params_desc, data):
    """
    Assumes every parameter is indepedent of each other
    func must be of signature (data, params_desc) -> float
    """
    print("Data size: {}".format(len(data)))
    orig = params_desc.copy()
    for k in params_desc:
        print("--------------------------")
        print("Parameter: {!r} ({})".format(k, params_desc[k]))
        if len(params_desc[k])==1: continue
        for kprime in params_desc:
            if (k != kprime) and len(params_desc[kprime])>1:
                params_desc[kprime] = [params_desc[kprime][0]]
        with Pool(8) as p:
            g = list(ParameterGrid([params_desc]))
            res = dict(zip([x[k] for x in g], p.map(func, g)))
        print(res)
        orig[k] = [max(res.keys(), key=lambda x: res[x])]
        params_desc = orig.copy()
    return orig
