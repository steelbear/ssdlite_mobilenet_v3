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
    file_list = glob.glob(os.path.join(args.dir, '*.json'))
    with open(args.output, 'w') as f:
        for file_path in file_list:
            dir_path, file_name = os.path.split(file_path)
            _, dir_name = os.path.split(dir_path)
            f.write(f'{dir_name}/{file_name[:-5]}\n')