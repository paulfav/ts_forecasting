#!/bin/bash

# Define the models, targets, and features arrays
models=("Autoformer" "Informer" "Transformer" "iTransformer" "Reformer")
targets=("close_AAPL" "close_AMZN" "close_MSFT" "close_META" "close_GOOGL" "close_NVDA" "close_SPY" "close_GM")
features=("S" "MS")
pred_lens=(1 10 50)
seq_lens=(10 30 50)
label_lens=(5 15 25)
max_jobs=4 # Maximum number of parallel jobs

# Function to ensure max parallel jobs
function job_control {
    while [ $(jobs -r | wc -l) -ge $max_jobs ]; do
        sleep 1
    done
}

# Loop over each model, target, and feature set
for model in "${models[@]}"; do
  for target in "${targets[@]}"; do
    for feature in "${features[@]}"; do
      for idx in "${!pred_lens[@]}"; do  # Loop through indices of pred_lens
        pred_len="${pred_lens[$idx]}"
        seq_len="${seq_lens[$idx]}"
        label_len="${label_lens[$idx]}"
        
        job_control  # Wait if we have reached max jobs

        if [ "$feature" = "S" ]; then
          # If features are S, set enc_in and dec_in to 1
          python -u run.py --model $model --data $target --data_path close_data.csv --features $feature --dec_in 1 --enc_in 1 --c_out 1 --target $target --pred_len $pred_len --train_epochs 30 --seq_len $seq_len --label_len $label_len --freq h --use_gpu False &
        else
          # If features are MS, set enc_in, dec_in, and c_out to 8
          python -u run.py --model $model --data $target --data_path close_data.csv --features $feature --dec_in 8 --enc_in 8 --c_out 8 --target $target --pred_len $pred_len --train_epochs 30 --seq_len $seq_len --label_len $label_len --freq h --use_gpu False &
        fi
      done
    done
    wait  # Wait for all processes of this target to finish before starting a new target
  done
done

wait  # Ensure that all background jobs finish before script exits

