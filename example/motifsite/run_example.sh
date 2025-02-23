#!/bin/bash

# please edit this line to your installation path
export MOTIFGEN='/home/hpark/programs/MotifGen'
prefix="148lE"

#step0: featurize
> python $MOTIFGEN/featurize/featurize_usage_latest.py $prefix.pdb
#output: prefix.prop.npz , prefix.lig.npz

#step1: motifnet -- run this on GPU!!!
# please check the environment 
> python $MOTIFGEN/predict.py $prefix.lig.npz

#step2: site & mol property predictor > {prefix}.log
> python $MOTIFGEN/Site/site_predictor.py $prefix
