DATASET=$1
CKPT_PATH=$2
OUTPUT_DIR=$3

python scripts/inference.py \
--ckpt-path $CKPT_PATH \
--dataset $DATASET \
--group-camera-name \
--output-dir $OUTPUT_DIR \
--block-size 4

python scripts/gather_inference_results.py \
--input-dir $OUTPUT_DIR \
--output-dir $OUTPUT_DIR/all

python scripts/save_parameters.py \
--dataset $DATASET \
--group-camera-name \
--output-dir $OUTPUT_DIR/all

python scripts/metrics.py \
--input-dir $OUTPUT_DIR/all