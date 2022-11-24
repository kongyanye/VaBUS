import argparse
import os
from pathlib import Path

from utils.video_source import FromImgDir, FromRawVideo


def generate_OD(video, model_type, resize, num, save_dir, overwrite,
                video_type):
    """generate inference results for Object Detection model (yolov3)"""

    if video == 'youtube':
        video_list = list(Path('../dataset/youtube').glob('*.mp4'))
    elif video == 'virat':
        video_list = list(Path('../dataset/VIRAT').glob('*.mp4'))
    else:
        video = Path(video)
        if video_type == 'video':
            if video.is_dir():
                video_list = list(video.glob('*.mp4'))
                print(
                    f'found {len(video_list)} videos under directory {video}')
            elif video.is_file():
                assert video.suffix in [
                    '.mp4', '.ts'
                ], ('invalid video suffix'
                    f'{video.suffix} for {str(video)} found')
                video_list = [video]
            else:
                raise Exception(f'unkown video source {video}')
        elif video_type == 'img':
            video_list = [video]
        else:
            raise Exception(
                f'unkown video_type {video_type}, must be one of [img, video]')

    for i, each in enumerate(video_list):
        print(f'[{i:04d}] {each}')

    assert model_type in ['yolo', 'd0', 'd3']
    if model_type == 'yolo':
        from models.yolov3_trt.yolov3 import Yolov3
        model = Yolov3()
    elif model_type == 'd0':
        from models.efficientdet import efficientdet
        model = efficientdet.TensorRTInfer(
            '/home/sig/files/VaBUS/src/models/efficientdet/efficientdet-d0.trt'
        )
    elif model_type == 'd3':
        from models.efficientdet import efficientdet
        model = efficientdet.TensorRTInfer(
            '/home/sig/files/VaBUS/src/models/efficientdet/efficientdet-d3.trt.fp16'
        )

    for video_path in video_list:
        if video_type == 'video':
            dataset = FromRawVideo(src_file=str(video_path), resize=resize)
        else:
            dataset = FromImgDir(dst_dir=str(video_path),
                                 resize=resize,
                                 filetype=args.img_type)
        if resize is None:
            resize_type = 'rawsize'
        elif isinstance(resize, list):
            resize_type = f"resize_{'_'.join([str(each) for each in resize])}"
        else:
            resize_type = f'resize_{resize}'
        output_file = Path(save_dir) / Path(
            f'OD_{resize_type}/{video_path.name}.csv')

        if output_file.is_file() and not overwrite:
            print(f'file {str(output_file)} already exists, skipping...')
            continue
        output_file.parent.mkdir(exist_ok=True)

        print(f'generating object detection results for {str(video_path)}')
        with open(str(output_file), 'w') as f:
            f.write('ind,label,conf,x1,y1,x2,y2' + '\n')
            for ind, img in dataset.read(num=num):
                res = model.infer(img)
                for label, conf, (x1, y1, x2, y2) in res:
                    f.write(f'{ind},{label},{conf},{x1},{y1},{x2},{y2}' + '\n')


def generate_KD(video, model_type, resize, num, save_dir, overwrite):
    """generate inference results for Human Keypoint Detection
    model (OpenPifPaf)"""
    from models.openpifpaf_trt.openpifpaf import Openpifpaf

    if video == 'human36m':
        video_list = list(Path('../dataset/human36m').glob('*'))
    elif video == 'mupots3d':
        video_list = list(Path('../dataset/mupots3d').glob('*'))
    else:
        video = Path(video)
        assert video.is_dir(), f'{video} is not a valid image dir'
        video_list = [video]

    for i, each in enumerate(video_list):
        print(f'[{i:04d}] {each}')

    assert model_type in ['openpifpaf']
    model = Openpifpaf()

    for video_path in video_list:

        dataset = FromImgDir(dst_dir=str(video_path), resize=resize)

        if resize is None:
            resize_type = 'rawsize'
        elif isinstance(resize, list):
            resize_type = f"resize_{'_'.join([str(each) for each in resize])}"
        else:
            resize_type = f'resize_{resize}'
        output_file = Path(save_dir) / Path(
            f'KD_{resize_type}/{video_path.name}.csv')

        if output_file.is_file() and not overwrite:
            print(f'file {str(output_file)} already exists, skipping...')
            continue
        output_file.parent.mkdir(exist_ok=True)

        print('generating human keypoint detection results '
              f'for {str(video_path)}')
        with open(str(output_file), 'w') as f:
            f.write('ind,human_ind,kp_ind,x,y,conf' + '\n')
            for ind, img in dataset.read(num=num):
                res = model.infer(img)
                for human_ind, pred in enumerate(res):
                    dic = {}
                    for (a, b, c, d) in pred.decoding_order:
                        dic[a] = c
                        dic[b] = d
                    for kp_ind in dic:
                        x, y, conf = dic[kp_ind]
                        f.write(f'{ind},{human_ind},{kp_ind},{x},{y},{conf}' +
                                '\n')
    os._exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate inference results')
    parser.add_argument('--task', type=str, required=True, help='OD or KD')
    parser.add_argument('--model_type',
                        type=str,
                        required=True,
                        help='yolo, d0, d3 or openpifpaf')
    parser.add_argument('--dataset',
                        type=str,
                        nargs='+',
                        required=True,
                        help='the dataset to use, can be one of '
                        '[youtube, virat, human36m, mupots3d], '
                        'or path to a single video/directory')
    parser.add_argument('--resize',
                        nargs='+',
                        default=[406, 720],
                        type=int,
                        help='resize video frames')
    parser.add_argument('--num',
                        type=int,
                        required=True,
                        help='number of images to infer, -1 for all')
    parser.add_argument('--res_dir',
                        type=str,
                        default='../dataset/ground_truth',
                        help='the result dir to save')
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='whether to overwrite'
                        'existing results')

    parser.add_argument('--video_type',
                        type=str,
                        default='video',
                        help='the type of the video source, video or img')
    parser.add_argument('--img_type',
                        type=str,
                        default='jpg',
                        help='the img format to look for')

    args = parser.parse_args()
    assert isinstance(args.resize, list) and len(
        args.resize) == 2, 'please specify both width and height in integers'
    if args.resize == [-1, -1]:
        args.resize = None

    if args.task == 'OD':
        for dataset in args.dataset:
            generate_OD(video=dataset,
                        model_type=args.model_type,
                        resize=args.resize,
                        num=args.num,
                        save_dir=args.res_dir,
                        overwrite=args.overwrite,
                        video_type=args.video_type)
    elif args.task == 'KD':
        for dataset in args.dataset:
            generate_KD(video=dataset,
                        model_type=args.model_type,
                        resize=args.resize,
                        num=args.num,
                        save_dir=args.res_dir,
                        overwrite=args.overwrite)
    else:
        raise Exception(f'expect task in [OD, KD], got {args.task}')
