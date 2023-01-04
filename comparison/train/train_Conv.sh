#!/bin/bash
#SBATCH --partition=GPU4A100                                  # select partion GPU
#SBATCH --nodes=1                                        # number of nodes requested by user
#SBATCH --gres=gpu:4                                     # use generic resource GPU, format: --gres=gpu:[n], n is the number of GPU card
#SBATCH --time=7-00:00:00                                # run time, format: D-H:M:S (max wallclock time)
python -u train_Conv.py