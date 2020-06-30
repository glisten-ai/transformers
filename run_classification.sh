#wandb init;

# Run training
python examples/text-classification/run_doordash.py \
  --data_dir="/home/transformers-public/data_dir/" \
  --model_name_or_path="bert-base-multilingual-cased" \
  --run_name="sarah-test4" \
  --max_seq_length=64 \
  --per_gpu_train_batch_size=32 \
  --do_train \
  --logging_steps=100 \
  --writing_dir="/home/transformers-public" \
  --max_steps=100000 \
  --save_steps=2000 \
  --tokenized_root_dir="/home/doordash-by-dish-no-and/title-tokenized/bert-base-multilingual-cased"

