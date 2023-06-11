set -e

# Download dataset and calc checksum
if [ -d $DATASET_ROOT_DIR ]; then
    echo "Dataset root directory: $DATASET_ROOT_DIR"
else
    echo "Fetching data from $DATA_BUCKET..."
    mkdir -p $DATASET_ROOT_DIR
    gcloud storage cp --recursive gs://"$DATA_BUCKET/gwhd_2021/*" "$DATASET_ROOT_DIR"
    echo "Data fetched"

    echo "Calculating checksum..."
    python3 data/scripts/data_integrity.py $DATASET_ROOT_DIR $DATASET_MD5
    echo "Checksum OK"
fi

# Prep yolo format
if [ -d $DATASET_ROOT_DIR/yolo-format-dataset ]; then
    echo "Yolo format dataset: "$DATASET_ROOT_DIR/yolo-format-dataset
else
    echo "Preparing yolo format dataset..."
    python3 data/scripts/gwhd_format_to_yolo.py
    echo "Yolo format dataset done"
fi

# Fetch yolov7 weights for transfer learning
if [ -f $YOLOV7_ROOT_DIR/weights/yolov7_training.pt ]; then
    echo "Yolov7 weights: $YOLOV7_ROOT_DIR/weights/yolov7_training.pt"
else
    echo "Fetching yolov7 weights..."
    mkdir -p $YOLOV7_ROOT_DIR/weights
    gcloud storage cp gs://"$DATA_BUCKET/yolov7_training.pt" "$YOLOV7_ROOT_DIR/weights/yolov7_training.pt"
    echo "Yolov7 weights fetched"
fi

# Generate yaml configuration file from env variables
echo "train: $DATASET_ROOT_DIR/yolo-format-dataset/train.txt
val: $DATASET_ROOT_DIR/yolo-format-dataset/val.txt
test: $DATASET_ROOT_DIR/yolo-format-dataset/test.txt

# number of classes
nc: 1

# class names
names: [ 'wheat-head' ]" > $PROJ_PATH/data/auto_gwhd_2021.yaml

cd $YOLOV7_ROOT_DIR

# TODO: Change weights and hyp to arg instead of hard coding and batch size to env variable
python train.py \
    --workers 8 \
    --device 0 \
    --batch-size 8 \
    --data $PROJ_PATH/data/auto_gwhd_2021.yaml \
    --img 1024 1024 \
    --cfg $PROJ_PATH/config/yolov7.yaml \
    --weights $YOLOV7_ROOT_DIR/weights/yolov7_training.pt \
    --name yolov7-custom \
    --hyp data/hyp.scratch.custom.yaml
