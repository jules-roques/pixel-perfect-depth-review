#!/bin/bash -l
#SBATCH -p mesonet 
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=256G
#SBATCH --account=m25146        # Your project account 
#SBATCH --job-name=ppd      # Job name
#SBATCH --output=output/%x_%j.out  # Standard output and error log
#SBATCH --error=output/%x_%j.err  # Activate your virtual environment

spack load cuda@12.6.2
source venv/bin/activate

python -u benchmark.py
