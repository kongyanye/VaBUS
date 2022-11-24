#!/usr/bin/bash

role=cloud
task=KD
dataset=../dataset/mupots3d/TS11

python run_experiments.py --task $task --role $role --dataset $dataset --res_dir ../results/resolution_203_360 --batch_size 15  --resize 203 360 run

python run_experiments.py --task $task --role $role --dataset $dataset --res_dir ../results/resolution_406_720 --batch_size 15  --resize 406 720 run

python run_experiments.py --task $task --role $role --dataset $dataset --res_dir ../results/resolution_812_1440 --batch_size 15  --resize 812 1440  run

python run_experiments.py --task $task --role $role --dataset $dataset --res_dir ../results/resolution_1080_1920 --batch_size 15  --resize 1080 1920  run
