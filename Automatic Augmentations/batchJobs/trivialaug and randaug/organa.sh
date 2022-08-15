#!/bin/bash
                           
#SBATCH --no-requeue
#SBATCH --partition=alpha
#SBATCH --nodes=1                   
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH --mincpus=1
#SBATCH --time=24:00:00                             
#SBATCH --job-name=organa
#SBATCH --mail-type=ALL
#SBATCH --mail-user=colin.simon@mailbox.tu-dresden.de
#SBATCH --output=output-%j.out
#SBATCH --error=error-%j.out

module --force purge                          				
module load modenv/hiera CUDA/11.3.1 GCC/11.2.0 Python/3.9.6

source lib.sh

create_or_reuse_environment

cd /scratch/ws/0/cosi765e-python_virtual_environment/BenchAugmentations/medmnist_standalone/

python train_and_eval_pytorch.py --download --num_epochs 100 --data_flag organamnist --augmentation trivialaugment
python train_and_eval_pytorch.py --download --num_epochs 100 --data_flag organamnist --augmentation randaugment