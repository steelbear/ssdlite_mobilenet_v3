import os
import glob
import argparse

parser = argparse.ArgumentParser(
    description='CirCor Dataset file listing'
)
parser.add_argument('--dir', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--anno-ext', type=str, choices=['json', 'tsv'], default='tsv')
args = parser.parse_args()

if __name__ == '__main__':
    if args.anno_ext == 'json':
        anno_ext = '.json'
    else:
        anno_ext = '.tsv'
    file_list = glob.glob(os.path.join(args.dir, '*' + anno_ext))
    with open(args.output, 'w') as f:
        for file_path in file_list:
            dir_path, file_name_ext = os.path.split(file_path)
            _, dir_name = os.path.split(dir_path)
            file_name, _ = os.path.splitext(file_name_ext)
            f.write(f'{dir_name}/{file_name}\n')