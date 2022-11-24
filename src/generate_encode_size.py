import argparse
from pathlib import Path

from utils.video_source import (FromImgDir, FromRawVideo,
                                compress_and_get_size_hw,
                                compress_and_get_size_iio)


def main(video,
         high_quality_enc,
         batch_size,
         num_batch,
         resize,
         save_dir,
         encode_func,
         overwrite=False,
         video_type='video'):
    """
    Args:
        video: can be one of video_dir or video_path, video_dir is the
            directory containing videos to check, video_path is a single
            video file path
    """
    assert encode_func in ['iio',
                           'hw'], ("encode func must be one of [iio, hw]"
                                   f"got {encode_func}")

    if video == 'youtube':
        video_list = list(Path('../dataset/youtube').glob('*.mp4'))
    elif video == 'virat':
        video_list = list(Path('../dataset/VIRAT').glob('*.mp4'))
    elif video == 'human36m':
        video_list = list(Path('../dataset/human36m').glob('*'))
    elif video == 'mupots3d':
        video_list = list(Path('../dataset/mupots3d').glob('*'))
    else:
        if video_type == 'video':
            video = Path(video)
            if video.is_dir():
                video_list = list(video.glob('*.mp4'))
                print(f'found {len(video_list)} videos under '
                      f'directory {video}')
            elif video.is_file():
                assert video.suffix in [
                    '.mp4', '.ts'
                ], ('invalid video suffix'
                    f'{video.suffix} for {str(video)} found')
                video_list = [video]
        elif video_type == 'img':
            video_list = [Path(video)]
        else:
            raise Exception(
                f'unkown video_type {video_type}, must be one of [img, video]')

    for i, each in enumerate(video_list):
        print(f'[{i:04d}] {each}')

    for video_path in video_list:
        if video_type == 'video':
            dataset = FromRawVideo(src_file=str(video_path), resize=resize)
        else:
            dataset = FromImgDir(dst_dir=video_path,
                                 resize=resize,
                                 filetype=args.img_type)

        if resize is None:
            resize_type = 'rawsize'
        elif isinstance(resize, list):
            resize_type = f"resize_{'_'.join([str(each) for each in resize])}"
        else:
            resize_type = f'resize_{resize}'
        output_file = Path(save_dir) / Path(
            f'encode_{resize_type}_batch_{batch_size}_'
            f'{encode_func}/{video_path.name}.csv')

        if output_file.is_file() and not overwrite:
            print(f'file {str(output_file)} already exists, skipping...')
            continue

        output_file.parent.mkdir(exist_ok=True)

        print(f'calculating encode size for {str(video_path)}')
        print(f'saving to {str(output_file)}...')
        with open(str(output_file), 'w') as f:
            f.write('ind,size' + '\n')
            for ind, imgs in dataset.read_batch(batch_size=batch_size,
                                                num_batch=num_batch):
                if encode_func == 'iio':
                    size = compress_and_get_size_iio(
                        imgs,
                        f'./tmp/tmp_{batch_size}.mp4',
                        input_channel='bgr')
                else:
                    size = compress_and_get_size_hw(
                        imgs,
                        f'./tmp/tmp_{batch_size}.mp4',
                        input_channel='bgr',
                        high_quality=high_quality_enc)
                if size <= 0:
                    raise Exception(
                        f'invalid size {size} found for ind {ind}, exit')
                f.write(f'{ind},{size}' + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate encoding results')
    parser.add_argument('--dataset',
                        type=str,
                        nargs='+',
                        required=True,
                        help='the dataset to use, can be one of '
                        '[youtube, virat, human36m, mupots3d], '
                        'or path to a single video/directory')
    parser.add_argument('--high_quality_enc',
                        action='store_true',
                        help='whether to use high quality encoders,'
                        'should be set True for KD and False for OD')
    parser.add_argument('--resize',
                        nargs='+',
                        default=[406, 720],
                        type=int,
                        help='resize video frames')
    parser.add_argument('--batch_size',
                        type=int,
                        required=True,
                        help='batch size to encode')
    parser.add_argument('--num',
                        type=int,
                        required=True,
                        help='number of images to infer, -1 for all')
    parser.add_argument('--save_dir',
                        type=str,
                        default='../dataset/ground_truth',
                        help='the result dir to save')
    parser.add_argument('--encode_func',
                        type=str,
                        default='hw',
                        help='encoding function to use, iio or hw')
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

    for video in args.dataset:
        main(video=video,
             high_quality_enc=args.high_quality_enc,
             batch_size=args.batch_size,
             num_batch=args.num,
             resize=args.resize,
             save_dir=args.save_dir,
             encode_func=args.encode_func,
             overwrite=args.overwrite,
             video_type=args.video_type)
