import numpy as np
import gensim
import sys

#### UNTESTED FILE WARNING ####
### the idea is the following but not yet tested 

def save_only_ngrams(model_in, model_out_fname):
    model_in.init_sims()
    hash2index = np.zeros(model_in.wv.bucket, dtype=int) - 1
    for k, v in model_in.wv.hash2index.items():
        hash2index[k] = v
    vectors_ngrams_norm = model_in.wv.vectors_ngrams_norm
    min_n_max_n = [model_in.wv.min_n, model_in.wv.max_n]
    
    np.savez_compressed(
        model_out_fname,
        hash2index=hash2index,
        vectors_ngrams_norm=vectors_ngrams_norm,
        min_max_n = min_n_max_n
    )

if __name__ == "__main__":
    USAGE = "$ python {} large_model_file_name [small_model_file_name]"\
            .format(sys.argv[0])
    if len(sys.argv) < 2:
        print(USAGE)
    input_model_name = sys.argv[1]
    output_model_name = sys.argv[2] if len(sys.argv) > 2 else \
                        input_model_name + '_compressed.npz'
    model_in = gensim.models.Word2Vec.load(input_model_name)
    print('Done loading the model: {}'.format(input_model_name))
    save_only_ngrams(model_in, output_model_name)
        
