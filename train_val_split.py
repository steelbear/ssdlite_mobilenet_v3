import argparse
import glob
import os
from pathlib import Path
import random
import shutil


parser = argparse.ArgumentParser(
    prog='Train / Validation spliter'
)
parser.add_argument('--dataset_root', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--val_rate', type=float)
args = parser.parse_args()


def save_file_list(dir, output):
    file_list = glob.glob(os.path.join(dir, '*.tsv'))
    with open(output, 'w') as f:
        for file_path in file_list:
            filepath = file_path[:-4]
            f.write(filepath + '\n')


def main():
    record_path = os.path.join(args.dataset_root, args.dataset)
    train_dir = Path(args.dataset_root, 'train2017')
    val_dir = Path(args.dataset_root, 'val2017')

    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    with open(record_path, 'r') as record_f:
        images = record_f.readlines()
        images = list(map(lambda s: s[:-1], images))
        num_images = len(images)

    val_indice = random.sample(list(range(num_images)), k=int(args.val_rate * num_images))
    train_val_tag = [True] * num_images

    for i in val_indice:
        train_val_tag[i] = False

    for i, is_train in enumerate(train_val_tag):
        img_src = os.path.join(args.dataset_root, images[i] + '.png')
        tsv_src = os.path.join(args.dataset_root, images[i] + '.tsv')
        if is_train:
            shutil.move(img_src, train_dir)
            shutil.move(tsv_src, train_dir)
        else:
            shutil.move(img_src, val_dir)
            shutil.move(tsv_src, val_dir)
    
    save_file_list(train_dir, os.path.join(args.dataset_root, 'CIRCOR_TRAIN'))
    save_file_list(val_dir, os.path.join(args.dataset_root, 'CIRCOR_VAL'))


if __name__ == '__main__':
    main()