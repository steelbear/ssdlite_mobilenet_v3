# $1 => data-path
# $2 => val-rate
# $3 => anno-ext

python list_file.py --dir "$1/data" --output "$1/DATA" --anno-ext $3
python train_val_split.py --data-path $1 --dataset DATA --val-rate $2 --anno-ext $3
python sound2coco.py --data-path $1 --dataset CIRCOR_TRAIN --output "$1/annotations/circor_train2017.json" --anno-ext $3
python sound2coco.py --data-path $1 --dataset CIRCOR_VAL --output "$1/annotations/circor_val2017.json" --anno-ext $3