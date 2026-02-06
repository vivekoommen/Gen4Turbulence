#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=180g
##SBATCH -p a6000-gcondo 
##SBATCH --nodelist=gpu2253
#SBATCH -A gk-h100-gcondo
#SBATCH  -p h100-gcondo 
##SBATCH --nodelist=gpu2701
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:2


#SBATCH -J hit_GCD_lf4
#SBATCH -o train_GenCFD.out
#SBATCH -e train_GenCFD.err
## get tunneling info

module load python/3.11.0s-ixrhc3q

source ~/apps/GenCFD/bin/activate

./scripts/train_hit3d.sh 