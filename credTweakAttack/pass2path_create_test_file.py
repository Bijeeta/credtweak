import random
import csv
import sys
import numpy as np

def test_samples_csv_gen(csv_path):
    '''
    This function generates pairs of passwods from a given csv file,
    assuming the first 2 columns are words
    '''
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            yield row[0], row[1]

def create_test_file(csv_path, num_samples=5000, prob=0.001):
    '''
    This function creates a test file format ("pass1<tab>pass2") from a csv dataset
    '''
    coin = [True, False]
    probs = [prob, 1 - prob]
    sample_gen = test_samples_csv_gen(csv_path)
    count = 0
    with open(csv_path[:-4] + '_test.txt', 'w') as tfile:
        for i, sample in enumerate(sample_gen):
            if (count >= num_samples):
                break
            if (np.random.choice(coin, 1, p=probs)[0]):
                tfile.write(sample[0] + '\t' + sample[1] + '\n')
                count += 1
    print("Created test file of {} samples in {}".format(count, csv_path[:-4] + '_test.txt'))

def main():
    if (len(sys.argv) < 2 or len(sys.argv) > 4):
        print("Arguments: 1. Path to csv file ('./fname.csv') 2. Number of Samples to generate 3. Probability of picking a sample")
        raise SystemExit("No path to csv file supplied. Exiting...")
    for arg in sys.argv:
        print(arg)
    path_to_csv = sys.argv[1]
    if (len(sys.argv) >= 3):
        num_samples = int(sys.argv[2])
    else:
        num_samples = 5000
    if (len(sys.argv) == 4):
        prob = float(sys.argv[3])
    else:
        prob = 0.001

    create_test_file(path_to_csv, num_samples, prob)


if __name__ == "__main__":
    main()
