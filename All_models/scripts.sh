export CUDA_VISIBLE_DEVICES=1


!/bin/bash
# Job name:
SBATCH --job-name=test
#
# Account:
SBATCH --account=paulfaverjon
#
# Partition:
SBATCH --partition=partition_name
#
# Wall clock limit:
SBATCH --time=00:30:

python -u run.py \
 --model Autoformer \
 --data Close \
 --data_path close_data.csv \
 --features S \
 --dec_in 1 \
 --enc_in 1 \
 --target close_AAPL \
 --freq h \
 --use_gpu False

python -u run.py --model Autoformer --data Close --data_path close_data.csv --features MS --dec_in 8 --enc_in 8 --c_out 8 --target close_AAPL --freq h --use_gpu False