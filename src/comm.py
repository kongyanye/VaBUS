import json
import logging
import os
import pickle
import queue
import shutil
import threading
import time
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
from flask import Flask, after_this_request, request, send_file
from PIL import Image

from utils.misc import timing
from utils.safe_read_write import (safe_compress_and_get_size_iio,
                                   safe_cv2_write, safe_f_write,
                                   safe_pickle_dump, safe_txt_write,
                                   safe_compress_and_get_size_hw)

logging.getLogger("urllib3").setLevel(logging.ERROR)


class COMM(ABC):
    """Communication interface between edge and cloud. Edge call the
    service deployed on cloud."""
    @abstractmethod
    def send_roi(self):
        """Send roi to edge"""
        raise NotImplementedError

    @abstractmethod
    def get_roi(self):
        """Get roi from cloud"""
        raise NotImplementedError

    @abstractmethod
    def send_infer_results(self):
        """Send inference results to edge"""
        raise NotImplementedError

    @abstractmethod
    def get_infer_results(self):
        """Get inference results from cloud"""
        raise NotImplementedError

    @abstractmethod
    def send_bg(self):
        """Send background image to edge"""
        raise NotImplementedError

    @abstractmethod
    def get_bg(self):
        """Get background image from cloud"""
        raise NotImplementedError


class Emulation(COMM):
    """Communication between edge and cloud by writing and reading files
    on disk"""
    def __init__(self, role, read_start=0, shared_dir='./shared'):
        assert role in ['cloud',
                        'edge'], f'role must be cloud or edge, got {role}'

        raise Exception('emulation mode is no longer supported, please use the'
                        'implementation mode instead')

        self.role = role
        # read start index
        self.read_start = read_start
        # home directory to share files
        self.shared_dir = Path(shared_dir)
        # directory to hold sent frames / videos
        self.dir_frames = self.shared_dir / 'frames'
        # directory to hold frames for background reconstruction
        self.dir_bg = self.shared_dir / 'bg'
        # directory to hold inference results
        self.dir_infer = self.shared_dir / 'infer_res'
        # directory to hold other miscellaneous results
        self.dir_misc = self.shared_dir / 'misc'

        # only creates shared directory on cloud side
        if self.role == 'cloud':
            if os.path.exists(self.shared_dir):
                shutil.rmtree(self.shared_dir)
            os.mkdir(self.shared_dir)
            self.dir_frames.mkdir(exist_ok=True)
            self.dir_bg.mkdir(exist_ok=True)
            self.dir_infer.mkdir(exist_ok=True)
            self.dir_misc.mkdir(exist_ok=True)
        elif self.role == 'edge':
            # wait cloud to be ready
            while True:
                if self.get_ready_signal():
                    break
                else:
                    time.sleep(1)

    @timing
    def send_roi(self, ind, roi):
        filename = self.dir_frames / f'{ind}.mp4'
        size = safe_compress_and_get_size_iio(roi,
                                              filename,
                                              input_channel='bgr')
        return size

    def get_roi(self):
        ind = self.read_start

        while True:
            filename = self.dir_frames / f'{ind}.mp4'
            if not filename.is_file():
                time.sleep(0.01)
            else:
                t1 = time.time()
                cap = cv2.VideoCapture(str(filename))
                imgs = []
                while True:
                    ret, img = cap.read()
                    if not ret:
                        break
                    imgs.append(img)
                t2 = time.time()
                yield ind, imgs, t2 - t1
                ind += 1
                # delete the temporary file after consuming it
                filename.unlink()

    def send_infer_results(self, ind, res):
        filename = self.dir_infer / f'{ind}.pkl'
        safe_pickle_dump(res, str(filename))

    def get_infer_results(self):
        ind = self.read_start

        while True:
            filename = self.dir_infer / f'{ind}.pkl'
            if not filename.is_file():
                time.sleep(0.01)
            else:
                with open(str(filename), 'rb') as f:
                    dic = pickle.load(f)
                yield ind, dic
                ind += 1
                # delete the temporary file after consuming it
                filename.unlink()

    def send_bg(self, bg):
        filename = self.dir_bg / 'bg.jpg'
        safe_cv2_write(str(filename), bg)

    def get_bg(self):
        filename = self.dir_bg / 'bg.jpg'
        if not filename.is_file():
            return None, None
        else:
            bg = cv2.imread(str(filename))
            size = os.path.getsize(str(filename))
            shutil.copy(str(filename), f'./shared/bg/bg_{time.time()}.jpg')
            filename.unlink()
            return bg, size

    def send_ready_signal(self):
        filename = self.dir_misc / 'ready.sig'
        filename.touch()

    def get_ready_signal(self):
        filename = self.dir_misc / 'ready.sig'
        if filename.is_file():
            filename.unlink()
            return True
        else:
            return False

    def send_finished_signal(self):
        filename = self.dir_misc / 'finished.sig'
        filename.touch()

    def get_finished_signal(self):
        filename = self.dir_misc / 'finished.sig'
        if filename.is_file():
            filename.unlink()
            return True
        else:
            return False


class Implementation(COMM):
    """Communication between edge and cloud over http"""
    def __init__(self,
                 role,
                 dataset_name='',
                 read_start=0,
                 hname='127.0.0.1',
                 port='5000'):
        assert role in ['cloud',
                        'edge'], f'role must be cloud or edge, got {role}'

        if role == 'cloud':
            self.dir = Path('./shared_cloud_%s' % port)
        elif role == 'edge':
            self.dir = Path('./shared_edge_%s' % port)
        self.dataset_name = dataset_name
        # read start index
        self.read_start = read_start
        # host IP to initialize the server
        self.hname = hname
        # server port
        self.port = port
        self.finished = False

        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        os.mkdir(self.dir)
        self.dir_frames = self.dir / 'frames'
        self.dir_frames.mkdir(exist_ok=True)
        self.dir_bg = self.dir / 'bg'
        self.dir_bg.mkdir(exist_ok=True)
        self.dir_misc = self.dir / 'misc'
        self.dir_misc.mkdir(exist_ok=True)

        if role == 'cloud':
            self.init_http_server()
            self.ready = False
            self.queue_infer = queue.Queue()
        elif role == 'edge':
            self.session = requests.Session()
            # wait cloud to be ready
            while True:
                status, dataset_name = self.get_ready_signal()
                if status:
                    break
                else:
                    time.sleep(1)
            assert dataset_name == self.dataset_name, (
                f'edge dataset name {self.dataset_name} is inconsistent with '
                f'cloud dataset name {dataset_name}')

    def init_http_server(self):
        app = Flask('VaBUS')

        @app.route("/")
        def index():
            return "Welcome to use VaBUS!"

        @app.route("/ready", methods=["GET"])
        def get_ready():
            return str(self.ready) + ', ' + self.dataset_name

        @app.route("/finished", methods=["POST"])
        def send_finished():
            finished = request.form['finished'] == 'True'
            self.finished = finished
            return 'ok'

        @app.route("/send_roi/<ind>", methods=["POST"])
        def send_roi(ind):
            file_data = request.files["video"]
            save_path = self.dir_frames / f'{ind}.mp4'
            safe_f_write(save_path, file_data.read())
            return 'ok'

        @app.route("/get_infer_res", methods=["GET"])
        def get_infer():
            res = self.queue_infer.get()
            return res

        @app.route("/get_bg", methods=["GET"])
        def get_bg():
            filename = self.dir_bg / 'bg.jpg'

            # delete bg.jpg after sending
            @after_this_request
            def delete(response):
                if filename.is_file():
                    filename.unlink()
                return response

            if not filename.is_file():
                return 'None', 204
            else:
                return send_file(filename, mimetype='image/jpg')

        @app.route("/get_roi_params", methods=["GET"])
        def get_roi_params():
            filename = self.dir_misc / 'roi_params.txt'

            # delete bg.jpg after sending
            @after_this_request
            def delete(response):
                if filename.is_file():
                    filename.unlink()
                return response

            if not filename.is_file():
                return 'None', 204
            else:
                return send_file(filename)

        t = threading.Thread(target=app.run,
                             kwargs={
                                 'debug': False,
                                 'host': '0.0.0.0',
                                 'port': self.port
                             })
        t.start()

    @timing
    def send_roi(self, task, ind, roi, hw_acc=False, roi_params=None):
        filename = self.dir_frames / f'{ind}.mp4'
        if roi is None:
            os.system(f'touch {filename}')
            size = 0
        else:
            if hw_acc:
                if task == 'KD':
                    high_quality = True
                else:
                    high_quality = False
                size = safe_compress_and_get_size_hw(roi,
                                                     filename,
                                                     input_channel='bgr',
                                                     tmp_dir=self.dir_frames,
                                                     roi_params=roi_params,
                                                     high_quality=high_quality)
            else:
                size = safe_compress_and_get_size_iio(roi,
                                                      filename,
                                                      input_channel='bgr')
        video_to_send = {'video': open(filename, 'rb').read()}
        _ = self.session.post(
            f'http://{self.hname}:{self.port}/send_roi/{ind}',
            files=video_to_send)
        filename.unlink()
        return size

    def get_roi(self):
        ind = self.read_start

        while True:
            filename = self.dir_frames / f'{ind}.mp4'
            if not filename.is_file():
                time.sleep(0.01)
            else:
                size = os.path.getsize(filename)
                if size == 0:
                    yield ind, [], 0, 0, []
                    ind += 1
                    filename.unlink()
                    continue
                t1 = time.time()
                cap = cv2.VideoCapture(str(filename))
                imgs = []
                while True:
                    ret, img = cap.read()
                    if not ret:
                        break
                    imgs.append(img)
                t2 = time.time()
                if len(imgs) != 15:
                    shutil.copy(filename, './check.mp4')
                mvs = []
                yield ind, imgs, t2 - t1, size, mvs
                ind += 1
                # delete the temporary file after consuming it
                filename.unlink()

    def send_infer_results(self, ind, res):
        self.queue_infer.put({'ind': ind, 'res': res})

    def get_infer_results(self):
        while True:
            response = self.session.get(
                f'http://{self.hname}:{self.port}/get_infer_res')
            response_json = json.loads(response.text)
            ind = response_json['ind']
            dic = response_json['res']
            yield ind, dic

    def send_bg(self, bg):
        filename = self.dir_bg / 'bg.jpg'
        safe_cv2_write(str(filename), bg)
        size = os.path.getsize(filename)
        return size

    def get_bg(self):
        response = self.session.get(f'http://{self.hname}:{self.port}/get_bg')
        if response.status_code != 200:
            return None, None
        else:
            size = int(response.headers['Content-Length'])
            bg = Image.open(BytesIO(response.content))
            bg = np.array(bg)[:, :, ::-1]  # change to np.array and bgr channel
            return bg, size

    def send_roi_params(self, line):
        filename = self.dir_misc / 'roi_params.txt'
        safe_txt_write(filename, line)
        size = os.path.getsize(filename)
        return size

    def get_roi_params(self):
        response = self.session.get(
            f'http://{self.hname}:{self.port}/get_roi_params')
        if response.status_code != 200:
            return None
        else:
            size = int(response.headers['Content-Length'])
            roi_params = response.text
            filename = self.dir_misc / 'roi_params.txt'
            safe_txt_write(filename, roi_params)
            return size

    def send_ready_signal(self):
        self.ready = True

    def get_ready_signal(self):
        try:
            response = self.session.get(
                f'http://{self.hname}:{self.port}/ready')
            status, dataset_name = response.text.split(', ')
            return status == 'True', dataset_name
        except requests.exceptions.ConnectionError:
            return False, ''

    def send_finished_signal(self):
        _ = self.session.post(f'http://{self.hname}:{self.port}/finished',
                              data={'finished': True})

    def get_finished_signal(self):
        return self.finished
