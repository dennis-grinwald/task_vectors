#!/bin/bash
#SBATCH --job-name=gf
#SBATCH --partition=gpu-5h
#SBATCH --gpus-per-node=40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G
#SBATCH --output=logs/job-%j.out

cp /home/space/datasets-sqfs/FLIIDNIID/Data.sqfs /tmp/

apptainer run -B /tmp/Data.sqfs:/cluster:image-src=/ --nv taskvectors.sif python get_features.py