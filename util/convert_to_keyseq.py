from word2keypress import Keyboard
KB = Keyboard()
import csv_parallel as csvp
import sys
import json

def pw2keyseq(upws):
    u, pws = upws
    return [(u, json.dumps(pws), json.dumps([KB.word_to_keyseq(w) for w in pws]))]
    
def main():
    if len(sys.argv) < 3:
        print("USAGE: python {} <csvin_file> <csvoutfile>".format(sys.argv[0]))
        peint("example: python convert_to_keyseq.py /hdd/c3s/data/cleaned_email_pass_ts.csv /hdd/c3s/data/cleaned_email_pass_ts.keyseq.csv")
    csv_in_file = sys.argv[1]
    csv_out_file = sys.argv[2]
    csvp.CHUNK_SIZE = int(1e5)  # 1 million
    csvp.parallel_run_csv2dataset(csv_in_file, csv_out_file, pw2keyseq)

if __name__ == "__main__":
    main()
