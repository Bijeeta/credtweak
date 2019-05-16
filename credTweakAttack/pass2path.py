from pass2path_rnn_tf import predict_list_pass2path, predict_pass2path ,train_pass2path, path2pass_run_testset
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
import argparse

def main():
    parser = argparse.ArgumentParser(description="train and predict using pass2path network")
    parser.add_argument("-t","---train", help="train pass2path_rnn_tf",
                    action="store_true")
    parser.add_argument("-p","--predict", help="predict using a trained pass2path_rnn_tf",
                    action="store_true")
    parser.add_argument("-x","--test", help="test using a trained pass2path_rnn_tf",
                    action="store_true")
    parser.add_argument("-d","--dataset", type=str, help="path to csv dataset file")
    parser.add_argument("-s","--step", type=int, help="display step to show training progress, default: 10")
    parser.add_argument("-e","--epochs", type=int, help="number of epochs to run, default: 80")
    parser.add_argument("-b","--batch", type=int, help="batch size, default: 50")
    parser.add_argument("-z","--size", type=int, help="rnn size, default: 128")
    parser.add_argument("-l","--layers", type=int, help="number of layers in each rnn, default: 3")
    parser.add_argument("-m","--embed", type=int, help="embedding size for words, deafult: 200")
    parser.add_argument("-w","--beam_width", type=int, help="beam width, number of predictions, default: 10")
    parser.add_argument("-i","--edit_distance", type=int, help="maximum edit distance to consider during training, default: 3")
    parser.add_argument("-r","--learning_rate", type=float, help="learning rate of the network, default: 0.001")
    parser.add_argument("-k","--keep_prob", type=float, help="keep probability = 1 - dropout probability, default: 0.8")
    parser.add_argument("-a","--password", type=str, help="predict passwords for this password, default: 'password'")
    args = parser.parse_args()
    if (args.train):
        if not args.dataset:
            parser.print_help()
            raise SystemExit("CSV Dataset file not specified")
        if (args.step):
            display_step = args.step
        else:
            display_step = 10
        if (args.epochs):
            epochs = args.epochs
        else:
            epochs = 80
        if (args.batch):
            batch_size = args.batch
        else:
            batch_size = 50
        if (args.size):
            rnn_size = args.size
        else:
            rnn_size = 128
        if (args.layers):
            num_layers = args.layers
        else:
            num_layers = 3
        if (args.embed):
            embed_size = args.embed
        else:
            embed_size = 200
        if (args.beam_width):
            beam_width = args.beam_width
        else:
            beam_width = 10
        if (args.edit_distance):
            edit_distance = args.edit_distance
        else:
            edit_distance = 3
        if (args.learning_rate):
            learning_rate = args.learning_rate
        else:
            learning_rate = 0.001
        if (args.keep_prob):
            keep_probability = args.keep_prob
        else:
            keep_probability = 0.8

        train_pass2path(ds_csv_path = args.dataset, display_step=display_step, epochs=epochs, batch_size=batch_size,
                       rnn_size=rnn_size, num_layers=num_layers, embed_size=embed_size,
                       beam_width=beam_width, edit_distance=edit_distance, learning_rate=learning_rate,
                       keep_probability=keep_probability)
    elif (args.predict):
        if (args.password):
            password = args.password
        else:
            password = "password"

        if (args.batch):
            batch_size = args.batch
        else:
            batch_size = 50
        if (args.beam_width):
            beam_width = args.beam_width
        else:
            beam_width = 10

        predict_pass2path(password, batch_size=batch_size, beam_width=beam_width)
    
    elif(args.test):
        if not args.dataset:
            parser.print_help()
            raise SystemExit("Test file not specified")
        if (args.batch):
            batch_size = args.batch
        else:
            batch_size = 50
        acc = path2pass_run_testset(args.dataset, batch_size=batch_size)
        print("Accuracy on test set: {}".format(acc))
    else:
        parser.print_usage()
    

if __name__ == "__main__":
    main()
