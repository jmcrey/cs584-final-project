export N=0
export CUDA_VISIBLE_DEVICES=$N
export TRAIN_FILE=$HOME/Documents/cs584-final-project/data/fever_train.txt
export VALIDATION_FILE=$HOME/Documents/cs584-final-project/data/fever_validation.txt
export OUTPUT_DIR=/run/media/jmack/ColdStorage/models/bart_large

echo "Model train file $TRAIN_FILE"
echo "Model validation file $VALIDATION_FILE"
echo "Model output directory $OUTPUT_DIR"
sleep 3

python run_mlm.py \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path facebook/bart-large \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --do_train \
    --do_eval \
    --line_by_line \
    --per_device_train_batch_size 5 \
    --per_device_eval_batch_size 5 \
    --max_train_samples 100000