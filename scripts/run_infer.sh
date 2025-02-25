#!/bin/sh
#SBATCH -J frozen_11
#SBATCH -p normal.q
#SBATCH -w star045
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -o ./logs/11.log
#SBATCH -e ./logs/11.logerr

python predict_pepscore.py --checkpoint ../params/pepscore_0.pt --input ../example/pepscore/ --emb_features 20 --esm_emb_features 16 --out_features 16 --attention_heads 2 --num_layers 1