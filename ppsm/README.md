# Personalized Password Strength Meter

## Train model

Input file - Csv file containing username and list of passwords

Usage - python train_fasttext:.py <train_file>

## PPSM

Usage - python ppsm.py <model_file> <input_password> <target_password>


## PPSM Compressed
`ppsm.py` file uses a compressed version of the model. Right now it only
considers the ngrams and not the vocabulary words. The compressed model file can
be found at [this
link](https://drive.google.com/file/d/1vJcBysoFNYnRr8QN3_eE7UKx5_GRxY5m/view?usp=sharing) (Size: 184 MB)

The file name is: `fastText2_keyseq_mincount:10_ngram:1-4_negsamp:5_subsamp:0.001_d:100_compressed.npz`.

```bash
$ python model_compress.py <large_model_file> <compressed_model_file>
```


### TODO 
* Add the product quantizer-based code to the repo. 
