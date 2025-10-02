pjsub --interact -L rscgrp=cx-interactive,jobenv=singularity,gpu=1
module load singularity
singularity shell --bind $HOME,/data/group1/${USER} --nv /data/group1/${USER}/latest.sif
