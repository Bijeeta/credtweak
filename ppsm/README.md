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
link](https://drive.google.com/file/d/1C4nceRFWKYCHUstEb_0sX4RRWaMcewkY/view) (Size: 192001572 bytes, 184 MB)

SHA2 hash: `544627bebc567a08917fb17e78f21c179e1d6108c0cdbb61b3a7d510f4678057`
The file name is: `fastText2_keyseq_mincount:10_ngram:1-4_negsamp:5_subsamp:0.001_d:100_compressed.npz`.

```bash
$ sha256sum fastText2_keyseq_mincount:10_ngram:1-4_negsamp:5_subsamp:0.001_d:100_compressed.npz
544627bebc567a08917fb17e78f21c179e1d6108c0cdbb61b3a7d510f4678057  fastText2_keyseq_mincount:10_ngram:1-4_negsamp:5_subsamp:0.001_d:100_compressed.npz
```


```bash
$ python model_compress.py <large_model_file> <compressed_model_file>
```


### TODO 
* Add the product quantizer-based code to the repo. 
