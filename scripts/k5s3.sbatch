#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=hpc4473C0
#SBATCH --output=outputs/k5s3.out

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate harienv

python ../main.py --config ../resnet_configs/kernel_config.yaml --resnet_architecture k5s3
