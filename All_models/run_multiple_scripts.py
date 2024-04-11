#this file will be used to run multiple scripts

import os
import subprocess
import time

import subprocess

def run_command():
    # Define the command to be executed
    command = """
    srun -A ic_fintech296mb -p savio2 -t 00:30:00 -N 1 python run.py -model Autoformer --data Close --data_path close_data.csv --features S --dec_in 1 --enc_in 1 --target close_AAPL --freq h --use_gpu True --seq_len 24 --label_len 12 --pred_len 1 --e_layers 2 --d_layers 1 --factor 1 --d_model 32 --n_heads 4 --d_ff 512 --train_epochs 10
    """
    
    # Execute the command
    try:
        subprocess.run(command, check=True, shell=True)
        print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        print("An error occurred while executing the command.")
        print(e)

if __name__ == "__main__":
    run_command()

    


