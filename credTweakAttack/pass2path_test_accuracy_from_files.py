# imports
import json
import numpy as np
import argparse

def test_accuracy_from_files_no_dict(tst_file, predictions_file, num_uniuqe_predictions=5, bias=False):
    '''
    This functions tests the accuracy of predictions made by Pass2Path and saved in a predictions file on
    the input test file.
    Inputs:
    tst_file - txt file in which every line is of the form: original_password<tab>target_password
    predictions_file - txt file in which every line is of the form: original_password<tab>json_list(prediction,score)
    num_uniuqe_predictions - number of uniuqe prediction to consider in the accuracy calculation. Predictions will be taken
                            in descending order of their score.
    This version does not use dictionary - number of lines in both files must match.
    Output: accuracy
    '''
    match_vec = []
    with open(tst_file, 'r') as f_tst:
        with open(predictions_file, 'r') as f_pred:
            for l, pred_line in enumerate(f_pred):
                orig_1, j_list = pred_line.split('\t')
                test_line = f_tst.readline()
                orig_2, target = test_line.split('\t')
                target = target.split('\n')[0]
#                 print(target)
                if (orig_1 != orig_2):
                    # Error
                    print("Error: Lines in file do not match. Stopping...")
                    break
                predictions_and_scores = json.loads(j_list)
                predictions = [pred[0] for pred in predictions_and_scores]
                # Take uniuqe predicions and add the original password as a
                # guess
                seen = set()
                seen_add = seen.add
                if (bias):
                    unq_predictions = [orig_1]
                    seen_add(orig_1)
                else:
                    unq_predictions = []
                unq_predictions += [x for x in predictions if not (x in seen or seen_add(x))]
                unq_predictions = unq_predictions[:num_uniuqe_predictions]
                if (target in unq_predictions):
                    match_vec.append(True)
                else:
                    match_vec.append(False)
        total_samples = l + 1
        if (total_samples != len(match_vec)):
            print("Error: Total tested samples ({}) does not match the number of lines in the files ({})"
                  .format(len(match_vec), total_samples))
        print("Accuracy calculated over {} samples".format(total_samples))
#         print(match_vec)
#         print(np.array(match_vec))
        acc = np.mean(np.array(match_vec))
        return acc
                    
def test_accuracy_from_files(tst_file, predictions_file, num_uniuqe_predictions=5, bias=False, write_pairs_to_file=False):
    '''
    This functions tests the accuracy of predictions made by Pass2Path and saved in a predictions file on
    the input test file.
    Inputs:
    tst_file - txt file in which every line is of the form: original_password<tab>target_password
    predictions_file - txt file in which every line is of the form: original_password<tab>json_list(prediction,score)
    num_uniuqe_predictions - number of uniuqe prediction to consider in the accuracy calculation. Predictions will be taken
                            in descending order of their score.
    This version uses dictionary - number of lines in both files may not match, accuracy is calculated on the
                                number o samples (lines) in the prediction file.
    Output: accuracy
    '''
    if (write_pairs_to_file):
        fout_name = predictions_file + ".pairs"
        print('Saving cracked pairs in {}'.format(fout_name))
        fout = open(fout_name, 'w')
    match_vec = []
    test_dict = {}
    hist_dict = {}
    with open(tst_file, 'r') as f_tst:
        for line in f_tst:
            splitted_line = line.split('\t')
            if (len(splitted_line) != 2):
                splitted_line = [s for s in splitted_line if s != '']
                orig = splitted_line[0]
                #target = splitted_line[-1]
                target = splitted_line[1:]
                #print(splitted_line)
                #print(target)
            else:
                orig, target = line.split('\t')
                target = [target]
            target[-1] = target[-1].split('\n')[0]
            if ('\n' in target):
                target = target[:-1]
            if (not test_dict.get(orig)):
                test_dict[orig] = target
            else:
                test_dict[orig] += target
            if (not hist_dict.get(orig)):
                hist_dict[orig] = 1
            else:
                hist_dict[orig] += 1
    
    with open(predictions_file, 'r') as f_pred:
        for l, pred_line in enumerate(f_pred):
                orig, j_list = pred_line.split('\t')
                targets = test_dict.get(orig)
                #print(targets)
                if (targets is None):
                    continue
                num_orig = hist_dict[orig]
                predictions_and_scores = json.loads(j_list)
                predictions = [pred[0] for pred in predictions_and_scores]
                # Take uniuqe predicions and add the original password as a
                # guess
                seen = set()
                seen_add = seen.add
                if (bias):
                    unq_predictions = [orig]
                    seen_add(orig)
                else:
                    unq_predictions = []
                unq_predictions += [x for x in predictions if not (x in seen or seen_add(x))]
                unq_predictions = unq_predictions[:num_uniuqe_predictions]
                #found_pred = False
                cracked = 0
                for t in targets:
                    if cracked < num_orig:
                        if (t in unq_predictions):
                            match_vec.append(True)
                            cracked += 1
                            if (write_pairs_to_file):
                                fout.write(orig + '\t' + t + '\t' + str(unq_predictions.index(t)) + '\n')   
                while (cracked < num_orig):
                    match_vec.append(False)
                    cracked += 1
                del test_dict[orig]
                    
    total_samples = len(match_vec)
    print("Accuracy calculated over {} samples".format(total_samples))
#         print(match_vec)
#         print(np.array(match_vec))
    acc = np.mean(match_vec)
    if (write_pairs_to_file):
        fout.close()
        print('Cracked pairs saved in {}'.format(fout_name))
    return acc

def main():
    parser = argparse.ArgumentParser(description="test pass2path network using test dataset and predictions file")
    parser.add_argument("-t","--test_file", type=str, help="path to test dataset file, tab separated")
    parser.add_argument("-p","--pred_file", type=str, help="path to predictions file, password<tab>json_list_of_predictions_and_scores")
    parser.add_argument("-n","--num_uniuqe", type=int, help="number of uniuqe predictions to consider in the accuracy calculation")
    parser.add_argument("-s","--save_pairs", help="save test cracked pairs in file", action="store_true")
    #parser.add_argument("-l","--list", help="target is a tsv list", action="store_true")
    args = parser.parse_args()

    if (args.save_pairs):
        acc = test_accuracy_from_files(tst_file=args.test_file, predictions_file=args.pred_file, num_uniuqe_predictions=args.num_uniuqe, bias=True, write_pairs_to_file=True)
    else:
        acc = test_accuracy_from_files(tst_file=args.test_file, predictions_file=args.pred_file, num_uniuqe_predictions=args.num_uniuqe, bias=True, write_pairs_to_file=False)
    print(acc)


if __name__ == "__main__":
    main()