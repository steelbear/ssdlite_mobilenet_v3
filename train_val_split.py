import argparse
import glob
import os
from pathlib import Path
import random
import shutil


parser = argparse.ArgumentParser(
    prog='Train / Validation spliter'
)
parser.add_argument('--data-path', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--val-rate', type=float)
parser.add_argument('--anno-ext', type=str, choices=['json', 'tsv'], default='tsv')
args = parser.parse_args()


def save_file_list(dir, output, anno_ext):
    file_list = glob.glob(os.path.join(dir, '*' + anno_ext))
    with open(output, 'w') as f:
        for file_path in file_list:
            dir_path, file_name_ext = os.path.split(file_path)
            _, dir_name = os.path.split(dir_path)
            file_name, _ = os.path.splitext(file_name_ext)
            f.write(f'{dir_name}/{file_name}\n')


def main():
    if args.anno_ext == 'json':
        anno_ext = '.json'
    else:
        anno_ext = '.tsv'

    record_path = os.path.join(args.data_path, args.dataset)
    train_dir = Path(args.data_path, 'train2017')
    val_dir = Path(args.data_path, 'val2017')

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
        img_src = os.path.join(args.data_path, images[i] + '.png')
        anno_src = os.path.join(args.data_path, images[i] + anno_ext)
        if is_train:
            shutil.move(img_src, train_dir)
            shutil.move(anno_src, train_dir)
        else:
            shutil.move(img_src, val_dir)
            shutil.move(anno_src, val_dir)
    
    save_file_list(train_dir, os.path.join(args.data_path, 'CIRCOR_TRAIN'), anno_ext)
    save_file_list(val_dir, os.path.join(args.data_path, 'CIRCOR_VAL'), anno_ext)


if __name__ == '__main__':
    main()