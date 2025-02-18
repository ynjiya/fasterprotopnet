#!/bin/bash
#testslurm
EXPERIMENT_RUN=$(python -c "import settings; print(settings.experiment_run)")
echo $EXPERIMENT_RUN
#SBATCH --output=slurm20-$EXPERIMENT_RUN-%j.out
# To receive email notifications
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yt2623

TERM=vt100
echo This is a training of BetterProtoPNet with the following settings:
echo "Experiment run: $EXPERIMENT_RUN"
which python
cat /vol/bitbucket/yt2623/iso/BetterProtoPNet/settings.py
echo This is $( /bin/hostname )
echo The current working directory is $( pwd )
/usr/bin/nvidia-smi
source /vol/bitbucket/yt2623/venv/bin/activate
echo Date: $( date ) 
echo "Starting training..."
python main.py
echo Date: $( date ) 
/usr/bin/nvidia-smi
echo "Training finished."