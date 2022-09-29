#!/bin/bash
                           
#SBATCH --no-requeue
#SBATCH --partition=alpha
#SBATCH --nodes=1                   
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --mincpus=1
#SBATCH --time=24:00:00                             
#SBATCH --job-name=RandAug_v3_100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevin.kirsten@mailbox.tu-dresden.de
#SBATCH --output=output-NoAug_100-%j.out

module --force purge                          				
module load modenv/hiera GCC/10.3.0 OpenMPI/4.1.1 TensorFlow  

source env.sh

create_new_environment $SLURM_JOB_ID

cd /home/keki996e/AL4ML/DATAAUG/Bench/TestTensorNotebook/v3_SimpleCNN/RandAug_V3

python LungTensorV3.py --epochs 100 --randaug --tag T1

remove_new_environment $SLURM_JOB_ID

