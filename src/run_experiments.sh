#!/usr/bin/bash

role=cloud
dataset='youtube virat'

# baseline
python run_experiments.py --task OD --role $role --dataset $dataset --res_dir ../results/baseline_OD --enable_bg False --enable_bg_overlay False --enable_aw False --enable_roienc False run

# enable_bg
python run_experiments.py --task OD --role $role --dataset $dataset --res_dir ../results/baseline_bg_OD --enable_aw False --enable_roienc False run

# enable_bg + enable_aw
python run_experiments.py --task OD --role $role --dataset $dataset --res_dir ../results/baseline_bg_aw_OD --enable_roienc False run

# enable_bg + enable_aw + enable_roienc
python run_experiments.py --task OD --role $role --dataset $dataset --res_dir ../results/baseline_bg_aw_roienc_OD run
