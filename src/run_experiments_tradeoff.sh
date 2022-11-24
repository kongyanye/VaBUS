#!/usr/bin/bash

role=cloud
dataset='virat'


for bw_limit in 300 600 900; do
    echo "bw_limit is ${bw_limit}Kbps"

    if [ $role == "cloud" ]; then
        sudo /home/sig/files/wondershaper/wondershaper -a enp0s31f6 -c
        sudo /home/sig/files/wondershaper/wondershaper -a enp0s31f6 -d $bw_limit -u $bw_limit
    fi

    echo "running baseline..."
    python run_experiments.py run --task OD --role $role --dataset $dataset --cloud_file cloud.py --edge_file edge.py --num 120 --res_dir ../results/tradeoff_baseline_${bw_limit} --enable_bg False --enable_bg_overlay False --enable_aw False --enable_roienc False

    echo "running vabus..."
    python run_experiments.py run --task OD --role $role --dataset $dataset --cloud_file cloud.py --edge_file edge.py --num 120 --res_dir ../results/tradeoff_vabus_${bw_limit}

    echo "running eaar"
    python run_experiments.py run --task OD --role $role --dataset $dataset --cloud_file cloud.py --edge_file edge_eaar.py --num 120 --res_dir ../results/tradeoff_eaar_${bw_limit}

    echo "running elf"
    python run_experiments.py run --task OD --role $role --dataset $dataset --cloud_file cloud_elf.py --edge_file edge_elf_qp.py --num 120 --res_dir ../results/tradeoff_elf_${bw_limit} --overwrite

    echo "running dds"
    python run_experiments_dds.py $role True ../results/tradeoff_dds_${bw_limit} $dataset

done

if [ $role == "cloud" ]; then
    echo futureissig | sudo -S /home/sig/files/wondershaper/wondershaper -a enp0s31f6 -c
fi
