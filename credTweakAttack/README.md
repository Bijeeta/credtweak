# pass2path
A variant of seq2seq Encoder-Decoder RNN model that learns pairs of
(password, transition path), where given a password and a transition path, a
new password is generated.

This model is based on JayPark's seq2seq model (Python 2): https://github.com/JayParks/tf-seq2seq

## Prerequisites
|Library         | Version |
|----------------------|----|
|`Python`|  `3.6.6 (Anaconda)`|
|`tensorflow`|  `1.10.0`|
|`word2keypress`|  `-`|
|`numpy`|  `1.14.5`|

## Data files

|File name         | Purpsoe |
|----------------------|----|
|`trans_dict_2idx.json`|  Transformation to an index.|

## File Formats

### Dictionary
A `Python` dictionary dumped into a `.json` file.

|Key       | Value |
|----------------------|----|
|Transition (e.g: `"('d', None, 0)"`)         | Index (e.g: `194`) |

The dictionary should be named `trans_dict_2idx.json`.

### Generating Paths
`.csv` file in the form:

|Column 1        | Column 2 |
|----------------------|----|
|`username/email/target`         | `list of passwords` |

### Training
`.csv` file (created using `Generating Paths`, see above) in the form:

|Column 1  | Column 2 | Column 3 |
|----------|----------|----------|
|source password (e.g:`Gameboy5`)| target password (e.g: `gamebo`) | path (e.g: `[0, 7, 8]`) |

### Testing

`.txt` file or just a regular file in a TSV (tab-seperated values) format, e.g:

`original password` `<tab>` `target password`


## Training Preparations

Before you start training the model, you should make sure:
1. Dictionary file is in `/data/` and the full path is: `/data/trans_dict_2idx.json`. You can generate this file on your own using the `generate_transition_dict` function in `edit_distance_backtrace.py`.
2. The dataset has been preprocessed such that for every user/mail/target's list of password in the original dataset there is now a `.csv` file with pair of passwords (souce,target) and the corresponding transition path. You can generate this file on your own using `generate_transition_dataset.py` as follows: `python generate_transition_dataset.py path/to/original_dataset.csv path/to/transition_dataset.csv`

## Main App:

You should only use the `pass2path_v2.py` file with the following arguments:

|Argument                 | Description                                 |
|-------------------------|---------------------------------------------|
|-h, --help       | shows arguments description             |
|-t, --train      | train pass2path model                   |
|-p, --predict    | predict using a trained pass2path model |
|-x, --test       | test (file) using a trained pass2path model |
|-q, --residual| use residual connections between layers (training) |
|-o, --save_pred| save predictions to a file when testing (testing) |
|-d, --dataset| path to a `.csv` dataset (training, testing)|
|-c, --cell_type| RNN cell type - `lstm` or `gru` (training, default: `lstm`) |
|-s, --step| display step to show progress (training, default: 100) |
|-e, --epochs| number of epochs to train the model (training, default: 80) |
|-b, --batch_size| batch size to draw from the provided dataset (training, testing, default: 50) |
|-z, --size| rnn size - number of neurons per layer (training, default: 128) |
|-l, --layers| number of layers in the network (training, default: 3) |
|-m, --embed| embedding size for sequences (training, default: 200) |
|-w, --beam_width| beam width, number of predictions (testing, predicting, default: 10) |
|-i, --edit_distance| maximum edit distance of password pairs to consider during training (training, default: 3) |
|-f, --save_freq| frequency to save checkpoints of the model (training, default: 11500) |
|-k, --keep_prob| keep probability = 1 - dropout probability (training, default: 0.8) |
|-a, --password| predict passwords for this password (predicting, default: "password") |
|-j, --checkpoint| model checkpoint number to use when testing (testing, predicting, default: latest checkpoint in model dir) |
|-u, --unique_pred| number of unique predictions to generate when predicting from file (predicting, default: `beam_width`) |

## Training

Examples to start training:

* Note: if there are checkpoints in the `/model/` dir, and the model parameters are the same, training will automatically resume from the latest checkpoint (you can choose the exact checkpoint number by editing the `checkpoint` file in the `/model/` dir with your favorite text editor).

`python pass2path_v2.py -t -q -d ./dataset_tr.csv -b 256 -i 3 -r 0.001 -k 0.6 -s 100 -e 3 -l 3`

`python pass2path_v2.py -t -d ./dataset_tr.csv -b 128 -i 2 -r 0.0003 -k 0.5 -s 1000 -e 3 -l 4 -c gru`

`python pass2path_v2.py -t -d ./dataset_tr.csv -b 50 -i 4 -r 0.001 -k 0.7 -s 1000 -e 10 -l 4 -z 100 -c gru`

`python pass2path_v2.py -t -d ./dataset_tr.csv -b 10 -i 3 -r 0.0001 -k 0.6 -s 100 -e 10 -l 3 -z 128 -m 150 -c lstm`

`python pass2path_v2.py -t -d ./dataset_tr.csv -b 10 -i 3 -r 0.0001 -k 0.6 -s 100 -e 10 -l 3 -z 128 -m 150 -c lstm -f 3000`

Model's checkpoints are saved in `/model/` dir.

## Testing

Examples for testing a trained model:

`python pass2path_v2.py -x -w 10 -b 500 -d ./test_files/dataset_ts.tsv`

`python pass2path_v2.py -x -w 100 -b 200 -j 1667500 -d ./test_files/dataset_ts.tsv`

`python pass2path_v2.py -x -o -w 1000 -b 20 -j 1801120 -d ./test_files/dataset_ts.txt` (saving `.predictions` file)

## Predicting

Examples for predicting using a trained model:

Online:

`python pass2path_v2.py -p -w 10 -j 1667500 -a P@SSword`

`python pass2path_v2.py -p -w 100 -a corectHorseBatterStapler`

From File:

`python pass2path_v2.py -p -b 20 -w 2000 -u 1000 -j 1667500 -d ./test_files/list_of_pws_to_generate_predictions.txt`

`python pass2path_v2.py -p -b 100 -w 1000 -u 500 -d ./test_files/list_of_pws_to_generate_predictions.txt`
