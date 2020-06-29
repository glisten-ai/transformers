wandb init;

# Run training
python examples/text-retrieval/run_retrieval.py \
  --data_dir="/home/transformers-public/data_dir/" \
  --model_name_or_path="bert-base-multilingual-cased" \
  --run_name="sarah_test2" \
  --max_seq_length_title=128 \
  --max_seq_length_category=10 \
  --datasets="google,google-rappi" \
  --per_gpu_train_batch_size=64 \
  --do_train \
  --logging_steps=200 \
  --writing_dir="/home/transformers-public" \
  --max_steps=100000 \
  --save_steps=1000 \
  --tokenized_root_dir="/home/sarahwooders_gmail_com/transformers/rappi-data/tokenized/" \
  --dataset="tokenized-doordash"
