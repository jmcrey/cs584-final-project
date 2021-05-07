export N=0
export CUDA_VISIBLE_DEVICES=$N
export TRAIN_FILE=$HOME/Documents/cs584-final-project/data/fever_train.txt
export MODEL_DIR=/run/media/jmack/ColdStorage/models/bart_large
export OUTPUT_DIR=/run/media/jmack/ColdStorage/models/bart_large_dev

echo "Model train file $TRAIN_FILE"
echo "Model validation file $VALIDATION_FILE"
echo "Model output directory $OUTPUT_DIR"
sleep 3

python run_mlm.py \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_DIR \
    --train_file $TRAIN_FILE \
    --do_train \
    --line_by_line \
    --per_device_train_batch_size 5 \
    --per_device_eval_batch_size 5