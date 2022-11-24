#!/usr/bin/bash

task=KD
role=cloud

python run_experiments.py --task $task --role $role --dataset human36m mupots3d --res_dir ../results/baseline_bg_KD --batch_size 15 --enable_bg run

python run_experiments.py --task $task --role $role --dataset human36m mupots3d --res_dir ../results/baseline_bg_aw_KD --batch_size 15 --enable_bg --enable_aw run

