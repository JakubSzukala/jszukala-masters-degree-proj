set -e
set -u

# Parse args
while getopts ":w:n:" opt; do
    case $opt in
        w) WEIGHTS="$OPTARG"
        ;;
        n) RUN_NAME="$OPTARG"
        ;;
        \?) echo "Invalid option -$OPTARG" >&2
        ;;
    esac
done

# TODO: Move download, yaml gen and prep yolo format to separate script, doubling of code

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

# Generate yaml configuration file from env variables
echo "train: $DATASET_ROOT_DIR/yolo-format-dataset/train.txt
val: $DATASET_ROOT_DIR/yolo-format-dataset/val.txt
test: $DATASET_ROOT_DIR/yolo-format-dataset/test.txt

# number of classes
nc: 1

# class names
names: [ 'wheat-head' ]" > $PROJ_PATH/data/auto_gwhd_2021.yaml

cd $YOLOV7_ROOT_DIR

# Is conf correct arg name?
python test.py \
    --data $PROJ_PATH/data/auto_gwhd_2021.yaml \
    --img 1024 \
    --batch 1 \
    --conf 0.001 \
    --iou 0.65 \
    --device 0 \
    --weights $WEIGHTS \
    --name $RUN_NAME