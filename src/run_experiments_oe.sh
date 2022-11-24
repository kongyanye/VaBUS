#!/usr/bin/bash

role=cloud
dataset=../dataset/youtube/crossroad2.mp4

python run_experiments.py --task OD --role $role --dataset $dataset --res_dir ../results/oe_oe_ind300 --enable_oe True --oe_method oe run

python run_experiments.py --task OD --role $role --dataset $dataset --res_dir ../results/oe_of_ind300 --enable_oe True --oe_method of run

python run_experiments.py --task OD --role $role --dataset $dataset --res_dir ../results/oe_worst_ind300 --enable_oe True --oe_method worst run

python run_experiments.py --task OD --role $role --dataset $dataset --res_dir ../results/oe_best_ind300 --enable_oe True --oe_method best run

python run_experiments.py --task OD --role $role --dataset $dataset --res_dir ../results/oe_nooe_ind300 --enable_oe False run
