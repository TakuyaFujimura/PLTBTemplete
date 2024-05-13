#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00


bash ./base.sh "simpleadamw"
