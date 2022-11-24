import os
import pickle
import tempfile
from contextlib import contextmanager

import cv2

from .video_source import compress_and_get_size_iio, compress_and_get_size_hw

# Context managers for atomic writes courtesy of
# http://stackoverflow.com/questions/2333872/atomic-writing-to-file-with-python


@contextmanager
def _tempfile(*args, **kws):
    """ Context for temporary file.

    Will find a free temporary filename upon entering
    and will try to delete the file on leaving

    Parameters
    ----------
    suffix : string
        optional file suffix
    """

    fd, name = tempfile.mkstemp(*args, **kws)
    os.close(fd)
    try:
        yield name
    finally:
        try:
            os.remove(name)
        except OSError as e:
            if e.errno == 2:
                pass
            else:
                raise e


@contextmanager
def open_atomic(filepath, *args, **kwargs):
    """ Open temporary file object that atomically moves to destination upon
    exiting.

    Allows reading and writing to and from the same filename.

    Parameters
    ----------
    filepath : string
        the file path to be opened
    fsync : bool
        whether to force write the file to disk
    kwargs : mixed
        Any valid keyword arguments for :code:`open`
    """
    fsync = kwargs.pop('fsync', False)

    with _tempfile(dir=os.path.dirname(filepath)) as tmppath:
        with open(tmppath, *args, **kwargs) as f:
            yield f
            if fsync:
                f.flush()
                os.fsync(f.fileno())
        os.rename(tmppath, filepath)


def safe_pickle_dump(obj, fname):
    with open_atomic(fname, 'wb') as f:
        pickle.dump(obj, f, -1)


def safe_cv2_write(fname, img):
    with _tempfile(dir=os.path.dirname(fname), suffix='.jpg') as tmppath:
        cv2.imwrite(tmppath, img)
        os.rename(tmppath, fname)


def safe_f_write(fname, bytes, suffix='.mp4'):
    with _tempfile(dir=os.path.dirname(fname), suffix=suffix) as tmppath:
        with open(tmppath, 'wb') as f:
            f.write(bytes)
        os.rename(tmppath, fname)


def safe_txt_write(fname, lines):
    with _tempfile(dir=os.path.dirname(fname), suffix='.txt') as tmppath:
        with open(tmppath, 'w') as f:
            if isinstance(lines, str):
                f.writelines(lines)
            else:
                raise Exception('unknown lines type, require str'
                                f', got {type(lines)}')
        os.rename(tmppath, fname)


def safe_compress_and_get_size_iio(imgs,
                                   encoded_vid_path,
                                   input_channel='rgb'):
    with _tempfile(dir=os.path.dirname(encoded_vid_path),
                   suffix='.mp4') as tmppath:
        size = compress_and_get_size_iio(imgs, tmppath, input_channel)
        os.rename(tmppath, encoded_vid_path)
    return size


def safe_compress_and_get_size_hw(imgs,
                                  encoded_vid_path,
                                  input_channel='bgr',
                                  tmp_dir='./tmp',
                                  roi_params=None,
                                  high_quality=False):
    with _tempfile(dir=os.path.dirname(encoded_vid_path),
                   suffix='.mp4') as tmppath:
        size = compress_and_get_size_hw(imgs, tmppath, input_channel, tmp_dir,
                                        roi_params, high_quality)
        os.rename(tmppath, encoded_vid_path)
    return size
