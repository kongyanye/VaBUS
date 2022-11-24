import argparse
import os
import shutil
import time
from pathlib import Path

import pandas as pd
import yaml

to_remove = []


def run(args):
    video_list = []
    for dataset in args.dataset:
        if dataset == 'youtube':
            video_dir = '../dataset/youtube'
            video_list += sorted(list(Path(video_dir).glob('*.mp4')))
        elif dataset == 'virat':
            video_dir = '../dataset/VIRAT'
            video_list += sorted(list(Path(video_dir).glob('*.mp4')))[::-1]
        elif dataset in ['human36m', 'mupots3d']:
            video_list += sorted(list(Path(f'../dataset/{dataset}').glob('*')))
        else:
            video_list.append(Path(dataset))

    if len(video_list) == 0:
        print(f'no videos found for dataset {args.dataset}')

    Path(args.res_dir).mkdir(exist_ok=True)
    shutil.copy('./run_experiments.py', f'{args.res_dir}/run_experiments.py')

    skipped = []
    for each in video_list:
        if each.stem in to_remove:
            continue

        param = {
            'task': args.task,
            'model_type': 'yolo',
            'video_path': str(each),
            'read_start': 0,
            'num_batch': args.num,
            'macro_size': 16,
            'thresh_diff_smooth': 30,
            'input_size': args.resize,
            'batch_size': args.batch_size,
            'fps': 15,
            'thresh_roi_black': 30,
            'run_type': 'implementation',
            'hname': '10.0.0.1',
            'port': 5000,
            'iou_th': 0.5,
            'oks_th': 0.5,
            'ground_truth_avail': True,
            'encode_hardware_acceleration': True,
            'enable_bg_sub': args.enable_bg,
            'enable_bg_overlay': args.enable_bg_overlay,
            'enable_aw': args.enable_aw,
            'aw_ratio': args.aw_ratio,
            'enable_oe': args.enable_oe,
            'show_edge': False,
            'show_cloud': False,
            'enable_roi_enc': args.enable_roienc,
            'res_dir': args.res_dir,
            'save_intermediate': False,
            'oe_method': args.oe_method,
        }

        param_dir = Path(args.res_dir) / 'param'
        param_dir.mkdir(exist_ok=True)
        param_path = param_dir / f'{each.stem}.yml'
        with open(param_path, 'w') as f:
            yaml.dump(param, f)

        savepath = Path(f"{args.res_dir}/perf_{each.stem}.mp4.csv")
        if savepath.exists() and not args.overwrite:
            print(f'results {str(savepath)} already exist, skipping...')
            skipped.append(each)
        else:
            if args.role == 'cloud':
                run_file = args.cloud_file
            elif args.role == 'edge':
                run_file = args.edge_file
            else:
                raise Exception(f'role must be cloud or edge, got {args.role}')

            shutil.copy(f'./{run_file}', f'{args.res_dir}/{run_file}')
            os.system(f'python {run_file} {param_path}')
            time.sleep(3)

    print('-' * 30)
    print(f'skipped: {skipped}')


def interpret_cloud(args):
    print('-'*30)
    res_list = sorted(list(Path(args.res_dir).glob('[!summary]*.csv')))
    if len(res_list) == 0:
        print(f'no results found under {args.res_dir}')
        return
    print(f'found {len(res_list)} csv files under {args.res_dir}')
    save_path = Path(args.res_dir) / 'summary.csv'
    print(f'writing results to {save_path}...')
    f = open(str(save_path), 'w')
    f.write('dataset,f1-score,precision,recall,bg_size,roi_size,param_size,'
            'total_size,compress,num_batches\n')
    all_f1 = []
    for each in res_list:
        if each.stem[5:-4] in to_remove:
            continue
        df = pd.read_csv(each)
        name = each.stem[5:].replace('.mp4', '')  # remove 'perf_' and '.mp4'
        mean_precision = df['precision'].mean()
        mean_recall = df['recall'].mean()
        mean_f1 = df['f1_score'].mean()
        all_f1.append(mean_f1)
        sum_bg = df['bg_size'].sum() if 'bg_size' in df.columns else 0
        sum_roi = df['sent_size'].sum()
        sum_param = df['roi_param_size'].sum(
        ) if 'roi_param_size' in df.columns else 0
        sum_raw = df['raw_size'].sum()
        compress_ratio = (sum_raw - sum_bg - sum_roi) / sum_raw
        f.write(
            f'{name},{mean_f1:.3f},{mean_precision:.3f},{mean_recall:.3f},'
            f'{int(sum_bg)},{int(sum_roi)},{int(sum_param)},{int(sum_raw)},'
            f'{compress_ratio:.3f},{len(df)}\n')
    f.close()
    print(f'mean f1-score: {sum(all_f1)/len(all_f1)*100:.2f}%')


def interpret_edge(args):
    print('-'*30)
    res_list = sorted(list(Path(args.res_dir).glob('[!summary]*.csv')))
    if len(res_list) == 0:
        print(f'no results found under {args.res_dir}')
        return
    print(f'found {len(res_list)} csv files under {args.res_dir}')
    save_path = Path(args.res_dir) / 'summary.csv'
    print(f'writing results to {save_path}...')
    f = open(str(save_path), 'w')
    f.write('dataset,afps,latency,num_batches,cpu,memory,gpu\n')
    all_latency = []
    for each in res_list:
        if each.stem[5:-4] in to_remove:
            continue
        df = pd.read_csv(each)
        name = each.stem[5:].replace('.mp4', '')  # remove 'perf_' and '.mp4'
        mean_afps = df['afps'].mean()
        mean_total = df['total'].mean()
        all_latency.append(mean_total)
        mean_cpu = df['CPU'].mean()
        mean_mem = df['Memory'].mean()
        mean_gpu = df['GPU'].mean()
        f.write(f'{name},{mean_afps:.2f},{int(mean_total)},{len(df)},'
                f'{mean_cpu:.2f},{mean_mem:.2f},{mean_gpu:.2f}\n')
    f.close()

    print(f'mean latency: {sum(all_latency)/len(all_latency):.2f}ms')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError(
            'Boolean value (True or False) expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='run experiments over a set of videos')
    parser.add_argument('function', type=str, help='which function to run')
    parser.add_argument('--task', type=str, help='OD or KD')
    parser.add_argument('--role',
                        type=str,
                        required=True,
                        help='edge or cloud')
    parser.add_argument('--dataset',
                        nargs='+',
                        type=str,
                        help='the dataset to use, can be one of'
                        ' [youtube, virat, human36m, mupots3d], '
                        'or path to a single video')
    parser.add_argument('--res_dir',
                        type=str,
                        required=True,
                        help='the result dir to save')
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='whether to overwrite'
                        'existing results')
    parser.add_argument('--aw_ratio',
                        type=float,
                        default=1,
                        help='aw_ratio in range [0, 1]')
    parser.add_argument('--num', type=int, help='num_batch', default=1200)
    parser.add_argument('--cloud_file', type=str)
    parser.add_argument('--edge_file', type=str)

    # temporary args
    parser.add_argument('--batch_size', type=int, default=15)
    parser.add_argument('--enable_bg', type=str2bool, default=True)
    parser.add_argument('--enable_bg_overlay', type=str2bool, default=True)
    parser.add_argument('--enable_aw', type=str2bool, default=True)
    parser.add_argument('--enable_roienc', type=str2bool, default=True)
    parser.add_argument('--enable_oe', type=str2bool, default=False)
    parser.add_argument('--oe_method', type=str)
    parser.add_argument('--resize', nargs=2, type=int, default=[406, 720])
    args = parser.parse_args()

    if args.function == 'run':
        run(args)
    elif args.function == 'interpret':
        if args.role == 'cloud':
            interpret_cloud(args)
        elif args.role == 'edge':
            interpret_edge(args)
    else:
        raise Exception('function argument must be one of [run, interpret]'
                        f', got {args.function}')
