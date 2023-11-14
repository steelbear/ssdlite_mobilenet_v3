import json

train_json_path = 'coco/annotations/instances_train2017.json'
train_json_path = 'coco/annotations/instances_val2017_ori.json'

# load json data
with open(train_json_path, "r") as json_file:
    json_data = json.load(json_file)

del json_data['info']
del json_data['licenses']
# print(json_data.keys())
json_data['images'] = json_data['images'][:1000]
for k, v in json_data.items():
    print(k, len(v))

for idx, i in enumerate(json_data['images']):
    print(i)
    new_data = {
        'file_name': json_data['images'][idx]['file_name'],
        'id': json_data['images'][idx]['id'],
        "height": json_data['images'][idx]['height'],
        "width": json_data['images'][idx]['height'],
    }
    json_data['images'][idx] = new_data

    # for adx, anno in enumerate(json_data['annotations']):
    #     print(anno)

# for idx, i in enumerate(json_data['annotations']):
#     del json_data['annotations'][idx]['segmentation']
#     del json_data['annotations'][idx]['area']
#     print(json_data['annotations'][idx])
#     # break

#
with open('coco/annotations/instances_val2017_1000.json', 'w') as outfile:
    json.dump(json_data, outfile, indent=4)