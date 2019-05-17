import gensim
from gensim.models import FastText
import json
import csv
import time
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from word2keypress import Keyboard
import math


homedir = Path(__file__).resolve().parent.parent
sys.path.append(str(homedir))
from util import csv_parallel as csvp

KB = Keyboard()
MODEL_FOLDER = Path('model/')

class MySentences(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        with open(self.fname) as f:
            csvf = csv.reader(f)
            t1 = time.time()
            for i, (u, pwslist, pwkeyseqlist) in enumerate(csvf):
                # pws = json.loads(pwlist)
                # pws = json.loads(pwkeyseqlist)
                if i % 1000000 == 0:
                    t2 = time.time()
                    print("Processed {} lines in {} sec".format(i, t2-t1))
                    t1 = t2
                pws = eval(pwkeyseqlist)
                yield pws

negative = 5
subsampling = 1e-3
min_count = 10
minn = 1  # minn
maxn = 4  # maxn
SIZE = 200  # dimension of the model 


def train_model(fname):
    sentences = MySentences(fname)
    t1 = time.time()
    model = FastText(sentences, size=SIZE, min_count=min_count, workers=12,
                     negative=negative, sample=subsampling, window=20,
                     min_n=minn, max_n=maxn)
    t2 = time.time()
    print("time taken: {}".format(t2-t1))
    model_name = 'fastText2_keyseq_mincount:{}_ngram:{}-{}_negsamp:{}_subsamp:{}'\
                 .format(min_count, minn, maxn, negative, subsampling)

    model.save(str(MODEL_FOLDER / model_name))
    print("model saved")
    return model, str(MODEL_FOLDER / model_name)


def test_distance(testf, modelfname):
    model = FastText.load(modelfname)
    testd = pd.read_csv(testf)
    testd['kw1'] = testd['0'].apply(KB.word_to_keyseq)
    testd['kw2'] = testd['1'].apply(KB.word_to_keyseq)
    testd['uncompressed'] = testd.apply(lambda x: model.similarity(x['kw1'], x['kw2']), axis=1)
    outf_name = testf.replace('.csv', '') + '_{}.wscore.csv'.format(Path(modelfname).name)
    testd.to_csv(outf_name)
    if 'grank' in testd:
        dd = testd[(testd.grank < 1000) & (testd.grank > -1)]
        g = dd.grank.apply(lambda x: int(math.log(x+1, 10)))
        print("Correlation between score and guess rank: {}".format(
            np.corrcoef(dd.uncompressed, g)[0, 1]
        ))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("USAGE: python {} <clean_pws_tr.csv>")
        print("Example: python train_fasttext.py /hdd/c3s/data/cleaned_email_pass_tr.keyseq.csv")
        exit(-1)
    train_file = sys.argv[1]
    model, mf = train_model(train_file)
    print(model, mf)
    #test_distance(test_file, mf)




