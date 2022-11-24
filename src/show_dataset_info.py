import argparse
from pathlib import Path

from ffprobe import FFProbe

to_remove = []


def show_video_info(dir_or_video_path):
    dir_or_video_path = Path(dir_or_video_path)
    if dir_or_video_path.is_file():
        flist = [dir_or_video_path]
    else:
        flist = sorted(dir_or_video_path.glob('*.mp4'))

    total_length = 0
    all_frame_size = {}
    all_frame_rate = {}
    total_frame = 0
    skipped = []

    for ind, each in enumerate(flist):
        if each.stem in to_remove:
            skipped.append(each.stem)
            continue

        info = FFProbe(str(each)).streams[0]

        # video duration
        length = info.duration
        total_length += float(length)

        # frame size
        frame_size = info.frame_size()
        all_frame_size[frame_size] = all_frame_size.get(frame_size, 0) + 1

        # frame rate
        frame_rate = info.framerate
        all_frame_rate[frame_rate] = all_frame_rate.get(frame_rate, 0) + 1

        # frame count
        frames = info.nb_frames  # frames()
        if frames == 'N/A':
            frames = int(float(length) * frame_rate)
        total_frame += int(frames)

        print(f'[{ind:04d}] {each.stem}{each.suffix} duration: {length} s, '
              f'{frame_size}, {frame_rate} fps, # of frames: {frames}')

    print('\n----------------------------\nSummary:')
    print(f'found {len(flist)} videos with {dir_or_video_path}')
    print(f'total length: {total_length:.2f} s')
    print(f'frame_size: {all_frame_size}')
    print(f'frame rate (fps): {all_frame_rate}')
    print(f'total frames counts: {total_frame}')
    print(f'skipped: {skipped}')


def show_img_info(img_path, img_type='jpg'):
    img_path = Path(img_path)
    total = 0
    video_count = 0
    for each in img_path.glob('*'):
        video_count += 1
        flist = sorted(each.glob(f'*.{img_type}'))
        print(f'{img_path / each}, {len(flist)} jpg images')
        total += len(flist)

    print('\n----------------------------\nSummary:')
    print(f'found {video_count} image dirs with {img_path}')
    print(f'total {img_type} images: {total}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='show information for given videos')
    parser.add_argument('source', type=str, help='which dataset source to use')
    args = parser.parse_args()

    if args.source == 'youtube':
        show_video_info('/home/sig/files/VaBUS/dataset/youtube')
    elif args.source == 'virat':
        show_video_info('/home/sig/files/VaBUS/dataset/VIRAT')
    elif args.source == 'human36m':
        show_img_info('/home/sig/files/VaBUS/dataset/human36m')
    elif args.source == 'mupots3d':
        show_img_info('/home/sig/files/VaBUS/dataset/mupots3d')
    else:
        if Path(args.source).is_file() or Path(args.source).is_dir():
            show_video_info(args.source)
        else:
            assert Path(args.source).exists(), f'unknown source {args.source}'
            show_img_info(args.source)
