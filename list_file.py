import os
import glob
import argparse

parser = argparse.ArgumentParser(
    description='CirCor Dataset file listing'
)
parser.add_argument('--dir', type=str)
parser.add_argument('--output', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    file_list = glob.glob(os.path.join(args.dir, '*.tsv'))
    with open(args.output, 'w') as f:
        for file_path in file_list:
            filepath = file_path[:-4]
            f.write(filepath + '\n')
            print(filepath)