import argparse
import csv
import json
import os
import re

import cv2

"""
{
    images: [
        {
            file_name,
            id,
            height,
            width
        }
    ],
    annotations: [
        {
            segmentation,
            area,
            iscrowd,
            image_id,
            bbox,
            category_id,
            id
        }
    ],
    categories: [
        {
            supercategory,
            id,
            name
        }
    ]
}
"""

POSITION_LIST = ['AV', 'PV', 'MV', 'TV', 'Phc']

CATEGORIES = [
    {
        'supercategory': 'Heart Beat',
        'id': 1,
        'name': 'S1'
    },
    {
        'supercategory': 'Heart Beat',
        'id': 2,
        'name': 'S2'
    }
]

parser = argparse.ArgumentParser(
    prog='SoundData to COCO Converter'
)
parser.add_argument('--data-path', type=str, help='path to dataset folder')
parser.add_argument('--dataset', type=str, help='dataset record file')
parser.add_argument('--output', type=str, help='path to json file to save the result')
parser.add_argument('--anno-ext', type=str, choices=['json', 'tsv'], default='tsv')


def filename2id(filename):
    patient_id, position, num = re.findall(r'(\d+)\_(AV|PV|MV|TV|Phc)\_?(\d)?', filename)[0]
    if num == '':
        num = '1'
    id = patient_id + str(POSITION_LIST.index(position)) + num
    return int(id)


def get_image_and_anno(file_path, anno_ext):
    anno_infos = []

    filename = file_path.split('/')[-1]
    image_path = file_path + '.png'
    anno_path = file_path + anno_ext
    image_id = filename2id(filename)

    image = cv2.imread(os.path.join(args.data_path, image_path))
    height, width, _ = image.shape
    image_info = {
        "file_name": filename + '.png',
        "id": image_id,
        "height": height,
        "width": width
    }

    with open(os.path.join(args.data_path, anno_path), 'r') as f:
        if anno_ext == '.json':
            anno_list = json.load(f)
        else:
            anno_list = csv.reader(f, delimiter='\t')

        id = 0
        
        for xmin, ymin, xmax, ymax, anno in anno_list:
            if anno_ext == '.tsv':
                xmin, ymin, xmax, ymax, anno = map(int, [xmin, ymin, xmax, ymax, anno])
            category_id = 1 if int(anno) == 1 else 2
            bbox_height = ymax - ymin
            bbox_width = xmax - xmin
            anno_infos.append({
                "area": bbox_height * bbox_width,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [
                    xmin,
                    ymin,
                    bbox_width,
                    bbox_height,
                ],
                "category_id": category_id,
                "id": image_id * 100 + id
            })
            id += 1

    return image_info, anno_infos

    
def main():
    images = []
    annotations = []

    if args.anno_ext == 'json':
        anno_ext = '.json'
    else:
        anno_ext = '.tsv'

    record_path = os.path.join(args.data_path, args.dataset)
    print(record_path)

    with open(record_path, 'r') as f:
        line = f.readline()
        while line != '':
            line = line[:-1]
            image_info, anno_infos = get_image_and_anno(line, anno_ext)
            images.append(image_info)
            annotations.extend(anno_infos)
            line = f.readline()
    
    dataset_json = {
        'images': images,
        'annotations': annotations,
        'categories': CATEGORIES
    }

    os.makedirs(os.path.split(args.output)[0], exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(dataset_json, f)


if __name__ == '__main__':
    args = parser.parse_args()
    main()