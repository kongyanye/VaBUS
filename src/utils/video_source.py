import os
import pathlib
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import imageio as iio
import numpy as np


class VideoSource(ABC):
    """Base class for read in video sources."""
    def __init__(self, resize=None):
        self.height, self.width = None, None
        self.resize = resize

    @abstractmethod
    def read(self):
        """Return an iterator of BGR images."""
        raise NotImplementedError

    @abstractmethod
    def read_batch(self):
        """Return an iterator of batch of BGR images."""
        raise NotImplementedError

    def resize_img(self, img, resize):
        """Resize image to target size"""
        if self.height is None or self.width is None:
            self.height, self.width = img.shape[:2]
            original_h, original_w = self.height, self.width
            if resize is not None:
                # resize to a specifc width and height
                if isinstance(resize, tuple) or isinstance(resize, list):
                    assert isinstance(resize[0], int) and isinstance(
                        resize[1], int), 'size must be int'
                    self.height = resize[0]
                    self.width = resize[1]
                # resize a specific ratio
                elif isinstance(resize, float):
                    assert resize > 0 and resize < 1, 'resize should be in '
                    f'(0, 1) when setting to a float, got {resize}'

                    self.width = int(self.width * resize)
                    self.height = int(self.height * resize)
                # resize to a specific width
                elif isinstance(resize, int):
                    assert resize % 2 == 0, f'when setting resize to a spe-\
                    cific width, it should be an even value, got {resize}'

                    ratio = resize / self.width
                    self.width = resize
                    self.height = int(self.height * ratio)
                    # ffmpeg don't allow frame size to be odd
                    if self.height % 2 == 1:
                        self.height += 1
            print(f'resize frames from {original_h, original_w} to '
                  f'{self.height, self.width}')
        img = cv2.resize(img, (self.width, self.height))
        return img


class FromImgDir(VideoSource):
    """Read images from a directory of files."""
    def __init__(self, dst_dir: str, filetype='jpg', **kwargs):
        """
        Args:
            dst_dir: target directory to read images
        """
        assert Path(dst_dir).is_dir(), (f'dst_dir {dst_dir} is'
                                        'not a valid image directory')

        dst_dir = Path(dst_dir)
        self.name = dst_dir.name
        self.flist = sorted(dst_dir.glob(f'*.{filetype}'))
        assert len(
            self.flist) > 0, f'no {filetype} images found under {dst_dir}'
        super().__init__(**kwargs)

    def __len__(self):
        return len(self.flist)

    def read(self, start=0, num=-1):
        """read `num` images from the video from start `start`

        Args:
            start (int): start frame to read, i.e., number of frames to skip,
                default 0 to read from the beginning
            num (int): number of frames to read, default -1 to read all

        Returns:
            (ind, imgs): ind (int) is the original index in the image
                directory, imgs (np.ndarray) is the images with shape [batch,
                height, width, channel]
        """
        assert start < len(self.flist), (
            f'start frame value {start} must '
            f'be smaller than total frame number {len(self.flist)}')

        if num == -1:
            end = len(self.flist)
        else:
            end = start + num
        for ind, each in enumerate(self.flist[start:end]):
            img = cv2.imread(str(each))
            if self.resize is not None:
                img = self.resize_img(img, self.resize)
            yield start + ind, img

    def read_batch(self, batch_size, start=0, num_batch=-1):
        """read `num_batch` batches of images from the directory from start `start`

        Args:
            batch_size (int): batch size to read
            start (int): start batches to read, i.e., number of batches to
                skip, default 0 to read from the beginning
            num_batch (int): number of batches to read, default -1 to read all

        Returns:
            (ind, imgs): ind (int) is the original batch index in the video,
                imgs (np.ndarray) is the images with shape [batch, height,
                width, channel]
        """
        assert isinstance(batch_size, int), ('batch_size must be '
                                             f'int, got {batch_size}')

        assert start * batch_size < len(self.flist), (
            f'start value {start*batch_size}({start}*{batch_size})'
            f' must be smaller than total frame number {len(self.flist)}')

        if num_batch == -1:
            batch_num_end = len(self.flist)
        else:
            batch_num_end = (start + num_batch) * batch_size
        for ind, s in enumerate(
                range(start * batch_size, batch_num_end, batch_size)):
            if s + batch_size > len(self.flist):
                break
            e = min(s + batch_size, len(self.flist))
            imgs = []
            for each in self.flist[s:e]:
                img = cv2.imread(str(each))
                if self.resize is not None:
                    img = self.resize_img(img, self.resize)
                imgs.append(img)
            yield start + ind, np.array(imgs)


def video2img(src_file: str, dst_dir: pathlib.PosixPath):
    """Transform a video file to images.

    Transform will be bypassed if the directory dst_dir already exists.

    Args:
        src_file (str): video filepath
        dst_dir (pathlib.PosixPath): destination folder path to save
            transformed images
    """
    src_file = Path(src_file)
    if not src_file.is_file():
        raise Exception(f"oops, video source {src_file} does not exist")

    if dst_dir.is_dir():
        print(f"directory {dst_dir} already exists, using old results")
        return
    else:
        dst_dir.mkdir(parents=True, exist_ok=True)

    img_path = str(dst_dir) + "/%010d.png"
    decoding_result = subprocess.run(
        ["ffmpeg", "-y", "-i", src_file, "-start_number", "0", img_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True)

    if decoding_result.returncode != 0:
        print(decoding_result.stdout)
        print(decoding_result.stderr)
        raise Exception(f"Decoding file {src_file} failed")


class FromVideoImg(FromImgDir):
    """Split a video to images and then read images from the directory."""
    def __init__(self, src_file: str, **kwargs):
        assert Path(src_file).is_file(), f'src_file {src_file} is not a video'
        src_file = Path(src_file)
        self.name = src_file.name
        dst_dir = src_file.parent / src_file.stem / 'src'
        video2img(src_file, dst_dir)
        super().__init__(dst_dir=dst_dir, **kwargs)


class FromRawVideo(VideoSource):
    """Read images directly from a single video file."""
    def __init__(self, src_file: str, **kwargs):
        """
        Args:
            src_file (str): path to video file
        """
        assert Path(src_file).is_file(), f'src_file {src_file} is not a video'
        self.cap = cv2.VideoCapture(src_file)
        self.name = Path(src_file).name
        super().__init__(**kwargs)

    def read(self, start=0, num=-1):
        """read `num` images from the video from start `start`

        Args:
            start (int): start frame to read, i.e., number of frames to skip,
                default 0 to read from the beginning
            num (int): number of frames to read, default -1 to read all

        Returns:
            (ind, imgs): ind (int) is the original index in the video,
                imgs (np.ndarray) is the images with shape [batch, height,
                width, channel]
        """
        # skip the first the `start` frames
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
        ind = -1

        while True:
            ind += 1
            ret, img = self.cap.read()
            if not ret:
                break
            if self.resize is not None:
                img = self.resize_img(img, self.resize)
            # if already read `num` images
            if ind == num:
                return start + ind, img
            else:
                yield start + ind, img

    def read_batch(self, batch_size, start=0, num_batch=-1):
        """read `num_batch` batches of images from the video from start `start`

        Args:
            batch_size (int): batch size to read
            start (int): start batches to read, i.e., number of batches to
                skip, default 0 to read from the beginning
            num_batch (int): number of batches to read, default -1 to read all

        Returns:
            (ind, imgs): ind (int) is the original batch index in the video,
                imgs (np.ndarray) is the images with shape [batch, height,
                width, channel]
        """
        # skip the first the `start` batch
        if start != 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start * batch_size - 1)
        ind = -1

        while True:
            imgs = []
            ind += 1
            # read a batch of images
            for _ in range(batch_size):
                ret, img = self.cap.read()
                if not ret:
                    if len(imgs) == batch_size:
                        return start + ind, np.array(imgs)
                    else:
                        return
                if self.resize is not None:
                    img = self.resize_img(img, self.resize)
                imgs.append(img)
            # if already read `num_batch` batches
            if ind == num_batch:
                return start + ind, np.array(imgs)
            else:
                yield start + ind, np.array(imgs)


class FromCamera(VideoSource):
    """Read images directly from a camera device."""
    def __init__(self, resize, fps=30, flip_method=2):
        if isinstance(resize, list) or isinstance(resize, tuple):
            self.height = resize[0]
            self.width = resize[1]
        elif isinstance(resize, float):
            assert resize > 0 and resize < 1, (
                "resize of float type must be in the range (0, 1), "
                f"got {resize}")
            self.width = int(1920 * resize)
            self.height = int(1080 * resize)
        else:
            raise Exception('resize must be one of list, tuple or float')

        self.name = 'camera'
        self.fps = fps
        assert fps <= 30, f'fps must be smaller than 30, got {fps}'
        self.stream = self.gstreamer_pipeline(capture_width=1920,
                                              capture_height=1080,
                                              display_width=self.width,
                                              display_height=self.height,
                                              framerate=fps,
                                              flip_method=flip_method)
        self.cap = cv2.VideoCapture(self.stream, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise Exception(
                f'unable to open camera with gstreamer {self.stream}')

    def gstreamer_pipeline(
        self,
        capture_width=3264,
        capture_height=2464,
        display_width=1280,
        display_height=720,
        framerate=30,
        flip_method=2,
    ):
        return ("nvarguscamerasrc ! "
                "video/x-raw(memory:NVMM), "
                "width=(int)%d, height=(int)%d, "
                "format=(string)NV12, framerate=(fraction)%d/1 ! "
                "nvvidconv flip-method=%d ! "
                "video/x-raw, width=(int)%d, height=(int)%d, "
                "format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink" % (
                    capture_width,
                    capture_height,
                    framerate,
                    flip_method,
                    display_width,
                    display_height,
                ))

    def read(self, num=-1, **kwargs):
        ind = -1

        while True:
            ind += 1
            ret, img = self.cap.read()
            if not ret:
                break
            # if already read `num` images
            if ind == num:
                return ind, img
            else:
                yield ind, img

    def read_batch(self, batch_size, num_batch=-1, **kwargs):
        """read `num_batch` batches of images from the video from start `start`

        Args:
            batch_size (int): batch size to read
            num_batch (int): number of batches to read, default -1 to read all

        Returns:
            (ind, imgs): ind (int) is the original batch index in the video,
                imgs (np.ndarray) is the images with shape [batch, height,
                width, channel]
        """
        ind = -1

        while True:
            imgs = []
            ind += 1
            # read a batch of images
            for _ in range(batch_size):
                ret, img = self.cap.read()
                if not ret:
                    return ind, np.array(imgs)
                imgs.append(img)
            # if already read `num_batch` batches
            if ind == num_batch:
                return ind, np.array(imgs)
            else:
                yield ind, np.array(imgs)


def compress_and_get_size(start_id, images_dir, encoded_vid_name):
    """Compress images to video using ffmpeg, read from image directy

    Args:
        start_id (int): start id to read
        images_dir (str): directory holding the images
        encoded_vid_name (str): output mp4 file path

    Returns:
        size (float): size of encoded mp4 file in bytes
    """
    encoded_vid_name = Path(images_dir) / encoded_vid_name
    encoding_result = subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-start_number",
        str(start_id), '-i', f"{images_dir}/%010d.png", "-vcodec", "libx264",
        "-g", "15", "-keyint_min", "15", "-pix_fmt", "yuv420p", "-frames:v",
        '15', "-r", "30", encoded_vid_name
    ],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)
    if encoding_result.returncode != 0:
        raise Exception(f'Encoding {encoded_vid_name} failed. stdout: \
            {encoding_result.stdout}, stderr {encoding_result.stderr}')
    size = os.path.getsize(encoded_vid_name)
    return size


def compress_and_get_size_iio(imgs, encoded_vid_path, input_channel='rgb'):
    """Compress images to video using imageio package, read from memory instead
    of disk

    Args:
        imgs (np.ndarray): numpy array of images with shape [batch, height,
            width, channel]
        encoded_vid_path (str): output mp4 file path
        input_channel (str): input channel order, default to rgb

    Returns:
        size (float): size of encoded mp4 file in bytes
    """
    if input_channel == 'bgr':
        imgs = imgs[:, :, :, ::-1]
    iio.mimwrite(encoded_vid_path, imgs, fps=30, macro_block_size=None)
    # codec='h264_nvmpi'
    size = os.path.getsize(encoded_vid_path)
    return size


def compress_and_get_size_hw(imgs,
                             encoded_vid_path,
                             input_channel='bgr',
                             tmp_dir='./tmp',
                             roi_params=None,
                             high_quality=False):
    """Compress images to video using hardware acceleration on Jetson

    Args:
        imgs (np.ndarray): batches of roi images
        encoded_vid_path (str): output mp4 file path
        tmp_dir (str): temporary dicrectory to save .rgb file
        input_channel (str): default is bgr (required by video_encode)
        high_quality (bool): whether to use high quality encoding as baseline,
            this has a big influence on KD inference, little influence on OD
            inference

    Returns:
        size (float): size of encoded mp4 file in bytes
    """
    if input_channel == 'rgb':
        imgs = imgs[:, :, :, ::-1]
    height, width, _ = imgs[0].shape

    tmp_dir = Path(tmp_dir)
    if not tmp_dir.exists():
        tmp_dir.mkdir(parents=True)

    tmp_file = f"tmp_file_{str(time.time()).split('.')[-1]}.bgr"
    bgr_path = tmp_dir / tmp_file
    imgs.tofile(str(bgr_path))

    if roi_params is None or not Path(roi_params).exists():
        if high_quality:
            res = subprocess.run([
                "video_encode",
                str(bgr_path),
                str(width),
                str(height), "bgr24", "H264", encoded_vid_path
            ],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True)
        else:
            res = subprocess.run([
                "video_encode",
                str(bgr_path),
                str(width),
                str(height), "bgr24", "H264", encoded_vid_path, "-MaxQpI",
                "51", "-MinQpI", "30", "-MaxQpP", "51", "-MinQpP", "30",
                "-MaxQpB", "51", "-MinQpB", "30"
            ],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True)
    else:
        if high_quality:
            res = subprocess.run([
                "video_encode",
                str(bgr_path),
                str(width),
                str(height), "bgr24", "H264", encoded_vid_path, "--eroi",
                "-roi",
                str(roi_params)
            ],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True)
        else:
            res = subprocess.run([
                "video_encode",
                str(bgr_path),
                str(width),
                str(height), "bgr24", "H264", encoded_vid_path, "--eroi",
                "-roi",
                str(roi_params), "-MaxQpI", "51", "-MinQpI", "30", "-MaxQpP",
                "51", "-MinQpP", "30", "-MaxQpB", "51", "-MinQpB", "30"
            ],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True)

    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise Exception(f'Decoding file video_encode {bgr_path} to '
                        f'{encoded_vid_path} failed')

    bgr_path.unlink()
    size = os.path.getsize(encoded_vid_path)
    return size
