#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH -c 2

./base.sh "threelayer"
