export N=0
export CUDA_VISIBLE_DEVICES=$N
export OUTPUT_DIR=$HOME/Documents/cs584-final-project/models/gpt2

echo "Model output directory $OUTPUT_DIR"

python run_clm.py \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path gpt2 \
    --dataset_name cc_news \
    --do_train \
    --do_eval \
    --validation_split_percentage 15 \
    --per_device_train_batch_size 5 \
    --per_device_eval_batch_size 5 \
    --max_train_samples 150000 \
    --overwrite_output_dir