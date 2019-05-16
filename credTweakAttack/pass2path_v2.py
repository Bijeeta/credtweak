from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="train, test and predict using pass2path network")
    parser.add_argument("-t", "---train", help="train pass2path_v2",
                        action="store_true")
    parser.add_argument("-p", "--predict", help="predict using a trained pass2path_v2",
                        action="store_true")
    parser.add_argument("-x", "--test", help="test using a trained pass2path_v2",
                        action="store_true")
    parser.add_argument("-q", "--residual", help="use residual connections between neurons",
                        action="store_true")
    parser.add_argument("-o", "--save_pred", help="save test predictions in a file",
                        action="store_true")
    parser.add_argument("-d", "--dataset", type=str,
                        help="path to csv dataset file")
    parser.add_argument("-c", "--cell_type", type=str,
                        help="RNN cell type: lstm or gru")
    parser.add_argument("-s", "--step", type=int,
                        help="display step to show training progress, default: 100")
    parser.add_argument("-e", "--epochs", type=int,
                        help="number of epochs to run, default: 80")
    parser.add_argument("-b", "--batch_size", type=int,
                        help="batch size, default: 50")
    parser.add_argument("-z", "--size", type=int,
                        help="rnn size, default: 128")
    parser.add_argument("-l", "--layers", type=int,
                        help="number of layers in each rnn, default: 3")
    parser.add_argument("-m", "--embed", type=int,
                        help="embedding size for words, deafult: 200")
    parser.add_argument("-w", "--beam_width", type=int,
                        help="beam width, number of predictions, default: 10")
    parser.add_argument("-i", "--edit_distance", type=int,
                        help="maximum edit distance to consider during training, default: 3")
    parser.add_argument("-f", "--save_freq", type=int,
                        help="frequency to save checkpoints of the model, default: 11500")
    parser.add_argument("-r", "--learning_rate", type=float,
                        help="learning rate of the network, default: 0.001")
    parser.add_argument("-k", "--keep_prob", type=float,
                        help="keep probability = 1 - dropout probability, default: 0.8")
    parser.add_argument("-a", "--password", type=str,
                        help="predict passwords for this password, default: 'password'")
    parser.add_argument("-j", "--checkpoint", type=int,
                        help="checkpoint number, default: latest checkpoint in model dir")
    parser.add_argument("-u", "--unique_pred", type=int,
                        help="number of unique predictions to generate when predicting from txt file")
    args = parser.parse_args()
    if (args.train):
        from pass2path_v2_train import train, set_train_flags
        if not args.dataset:
            parser.print_help()
            raise SystemExit("CSV Dataset file not specified")
        if (args.step):
            display_step = args.step
        else:
            display_step = 100
        if (args.epochs):
            epochs = args.epochs
        else:
            epochs = 10
        if (args.batch_size):
            batch_size = args.batch_size
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
            keep_probability = 0.6
        if (args.residual):
            use_res = True
        else:
            use_res = False
        if (args.cell_type and args.cell_type == 'gru'):
            cell_type = 'gru'
        else:
            cell_type = 'lstm'
        if (args.save_freq):
            save_freq = args.save_freq
        else:
            save_freq = 11500

        set_train_flags(cell_type=cell_type, hidden_units=rnn_size, num_layers=num_layers, embed_size=embed_size, path_to_ds=args.dataset,
                        use_residual=use_res, keep_prob=keep_probability, edit_distance=edit_distance,
                        lr=learning_rate, batch_size=batch_size, epochs=epochs, display_step=display_step, save_freq=save_freq)
        train()

    elif (args.predict):
        from pass2path_v2_decode import predict, predict_batch, set_decode_flags
        if (args.password):
            password = args.password
        else:
            password = "password"

        if (args.batch_size):
            batch_size = args.batch_size
        else:
            batch_size = 50
        if (args.beam_width):
            beam_width = args.beam_width
        else:
            beam_width = 10
        if (args.checkpoint):
            checkpoint = args.checkpoint
        else:
            checkpoint = -1
        if (args.unique_pred):
            unq_pred = args.unique_pred
        else:
            unq_pred = beam_width
        if (args.dataset):
            set_decode_flags(goal='decode', checkpoint=checkpoint, beam_width=beam_width,
                             decode_batch_size=batch_size, tst_file_path=args.dataset)
            predict_batch(num_uniuqe_predictions=unq_pred, bias=True)
        else:
            set_decode_flags(goal='predict', checkpoint=checkpoint,
                             beam_width=beam_width, decode_batch_size=batch_size)
            predict(original_pass=password)

    elif(args.test):
        from pass2path_v2_decode import decode, set_decode_flags
        if not args.dataset:
            parser.print_help()
            raise SystemExit("Test file not specified")
        if (args.batch_size):
            batch_size = args.batch_size
        else:
            batch_size = 50
        if (args.beam_width):
            beam_width = args.beam_width
        else:
            beam_width = 10
        if (args.save_pred):
            save_predictions = True
        else:
            save_predictions = False
        if (args.checkpoint):
            checkpoint = args.checkpoint
        else:
            checkpoint = -1

        set_decode_flags(goal='decode', checkpoint=checkpoint, beam_width=beam_width,
                         decode_batch_size=batch_size, tst_file_path=args.dataset)
        decode(write_to_file=save_predictions)

    else:
        parser.print_usage()


if __name__ == "__main__":
    main()
