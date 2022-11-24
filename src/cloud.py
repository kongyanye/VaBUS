import logging
import os
import queue
import sys
import threading
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import yaml

from background_reconstruction import BackgroundReconstructor
from comm import Emulation, Implementation
from models.yolov3_trt.yolov3 import Yolov3
# from models.efficientdet import efficientdet
from models.openpifpaf_trt.openpifpaf import Openpifpaf
from roi_params import RoiParams
from utils.misc import (add_boxes_to_img, add_kp_to_img, concat_image,
                        get_logger, nms_filter)
from utils.perf import GroundTruth, PerfMeter
from utils.sort import Sort

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress tf INFO and WARNING
warnings.filterwarnings('ignore', category=UserWarning)  # suppress UserWarning
logger = get_logger(name='cloud', level='debug')
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('openpifpaf').setLevel(logging.ERROR)
logging.getLogger('models.openpifpaf_trt').setLevel(logging.ERROR)
logging.getLogger('models.openpifpaf_trt.visualizer.base').setLevel(
    logging.ERROR)


class Cloud:
    def __init__(self, dataset_name, task, model_type, hw_acc, resize,
                 batch_size, iou_th, oks_th, run_type, thresh_roi_black,
                 macro_size, read_start, fps, show_res, ground_truth_avail,
                 enable_bg_sub, enable_bg_overlay, enable_roi_enc, res_dir,
                 save_intermediate, port):
        self.dataset_name = dataset_name
        self.task = task
        self.model_type = model_type
        self.hw_acc = hw_acc
        self.resize = resize
        self.batch_size = batch_size
        self.iou_th = iou_th
        self.run_type = run_type
        self.thresh_roi_black = thresh_roi_black
        self.macro_size = macro_size
        self.read_start = read_start
        self.fps = fps
        self.show_res = show_res
        self.ground_truth_avail = ground_truth_avail
        self.enable_bg_sub = enable_bg_sub
        self.enable_bg_overlay = enable_bg_overlay
        self.enable_roi_enc = enable_roi_enc
        self.role = 'cloud'
        self.res_dir = Path(res_dir)
        self.res_dir.mkdir(parents=True, exist_ok=True)
        self.save_intermediate = save_intermediate
        if self.save_intermediate:
            self.inter_dir = './tmp/intermediate'
            Path(self.inter_dir).mkdir(parents=True, exist_ok=True)

        logger.info('initializing cloud service...')

        if self.run_type == 'emulation':
            self.communicator = Emulation(role=self.role,
                                          read_start=read_start)
        elif self.run_type == 'implementation':
            self.communicator = Implementation(role=self.role,
                                               dataset_name=dataset_name,
                                               read_start=read_start,
                                               port=str(port))
        else:
            raise Exception(f'unkown run_type {self.run_type}')

        # load models
        if task == 'OD':
            assert self.model_type in ['yolo', 'd0', 'd3'], f'model_type must be yolo, d0, or d3, got {self.model_type}'
            if self.model_type == 'yolo':
                self.model = Yolov3(multithread=True, warm_up=True)
            elif self.model_type == 'd0':
                self.model = efficientdet.TensorRTInfer('/home/sig/files/VaBUS/src/models/efficientdet/efficientdet-d0.trt', True)
            elif self.model_type == 'd3':
                self.model = efficientdet.TensorRTInfer('/home/sig/files/VaBUS/src/models/efficientdet/efficientdet-d3.trt.fp16', True)

        elif task == 'KD':
            self.model = Openpifpaf()
        self.height, self.width = None, None  # resolution of all the frames

        # threshold
        self.confidence_threshold = 0.5
        self.max_area_threshold = 0.04

        # background reconstruction
        self.bg_recon = BackgroundReconstructor(thresh_roi_black)
        self.cache_bg = queue.Queue()

        # for background overlay
        self.mask = None
        self.background = None  # after overlay (for comparision)
        self.background_static = None  # without overlay (for recovering)
        self.overlay_res = {}  # detection results of overlay objects
        self.overlay_ind = {}  # last batch index with overlay info updated

        # object tracker
        self.tracker = Sort()

        # ROI params
        self.roi_param = None
        self.region_large = None
        self.region_small = None

        # cache
        self.cache_roi = queue.Queue()
        self.cache_render = queue.Queue()
        self.cache_show = queue.Queue()

        # ground truth evaluation
        if self.model_type == 'd0':
            ground_truth_dir = '../dataset/ground_truth_d0'
        elif self.model_type == 'd3':
            ground_truth_dir = '../dataset/ground_truth_d3'
        else:
            ground_truth_dir = '../dataset/ground_truth'
        if self.ground_truth_avail:
            self.gt = GroundTruth(task=self.task,
                                  vid_name=dataset_name,
                                  hw_acc=self.hw_acc,
                                  resize=self.resize,
                                  batch_size=batch_size,
                                  th_iou=iou_th,
                                  th_oks=oks_th,
                                  th_min_area=256,
                                  ground_truth_dir=ground_truth_dir)

        # evaluate performance
        self.perf = PerfMeter()

        # ready
        self.communicator.send_ready_signal()
        logger.info('cloud is ready')

    def overlay_background(self, ind, img, static_id, cur_track):
        # for display and evaluation purpose, should be the same
        # as the edge side
        if len(static_id) > 0 and self.background is not None:
            background = self.background.copy()

            for i in static_id:
                x1, y1, x2, y2, label = cur_track[i]
                # overlay the background with objects
                background[y1:y2, x1:x2] = img[y1:y2, x1:x2]
                self.overlay_res[i] = [label, x1, y1, x2, y2]
                self.overlay_ind[i] = ind
            self.background = background

    def infer(self):
        logger.info('listening for inference inputs...')
        last_track = None
        for batch_ind, frames, t, size, mvs in self.communicator.get_roi():
            # print(f'infer frame {batch_ind}...')
            if self.height is None or self.width is None:
                self.height, self.width = frames[0].shape[:2]

            # log size
            self.perf.set_value(batch_ind, 'sent_size', size, 'byte', 'sum')

            if size == 0:
                self.perf.set_value(batch_ind, 'raw_size',
                                    self.gt.get_gt_size(batch_ind), 'byte',
                                    'sum')
                self.perf.set_value(batch_ind,
                                    'precision',
                                    0,
                                    '',
                                    'mean',
                                    digit=3)
                self.perf.set_value(batch_ind,
                                    'recall',
                                    0,
                                    '',
                                    'mean',
                                    digit=3)
                self.perf.set_value(batch_ind,
                                    'f1_score',
                                    0,
                                    '',
                                    'mean',
                                    digit=3)
                dic = {
                    'res_batch': [[]],
                    'infer_time': -1,
                    'decode': 0,
                    'static_id': [],
                    'batch_track': [],
                    'mvs': []
                    }
                self.communicator.send_infer_results(batch_ind, dic)
                self.perf.print(batch_ind)
                continue

            # infer
            t1 = time.time()
            frames_infer = []
            masks = []
            for frame_ind, frame in enumerate(frames):
                if self.enable_bg_sub:
                    frame, mask_fg = self.complement_roi_with_bg(
                        frame, batch_ind * self.batch_size + frame_ind)
                    masks.append(mask_fg)
                frames_infer.append(frame)
            if self.task == 'OD':
                res_batch = []
                for frame in frames_infer:
                    res = self.model.infer(frame)
                    res_batch.append(res)
                res_batch_KD = []
            elif self.task == 'KD':
                res_batch_KD = self.model.infer_batch(frames_infer)
                res_batch = []
                for res_KD in res_batch_KD:
                    res = []
                    for kd in res_KD:
                        x, y, w, h = kd.bbox()
                        x, y, w, h = x / 641 * self.width, \
                            y / 369 * self.height, w / 641 * self.width, \
                            h / 369 * self.height
                        x, y, w, h = int(x), int(y), int(w), int(h)
                        res.append(['persons', 1, (x, y, x + w, y + h)])
                    res_batch.append(res)
            t2 = time.time()
            # print(f'inference time for {len(res_batch)} frames: '
            #       f'{(t2-t1)*1000:.2f} ms')

            # initialize roi params
            if self.roi_param is None:
                self.roi_param = RoiParams(self.height, self.width)

            # number of frames where the object is static for each track ID
            static_count = {}

            # postprocess infer results
            nres_batch = []
            cur_track = {}  # initialize
            batch_track = []  # track results for the whole batch (used for oe)
            for ind, each in enumerate(res_batch):
                nres = []
                for label, conf, (x1, y1, x2, y2) in each:

                    # # filter out large regions, they are not true objects
                    # if w * h > self.max_area_threshold:
                    #     continue

                    # # filter out low confidence regions
                    # if conf < self.confidence_threshold:
                    #     continue
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    nres.append([label, x1, y1, x2, y2])

                nres_batch.append(nres)

                # update ROI params
                if self.enable_roi_enc:
                    self.cache_roi.put({'batch_ind': batch_ind, 'nres': nres})

                # update object tracker
                if not nres:
                    continue

                # update with bbox in each frame
                labels = [each[0] for each in nres]
                nres = np.array([each[1:] for each in nres])
                track_bb_id = self.tracker.update(labels, nres).tolist()

                # transform list into dict
                cur_track = {}
                for [x1, y1, x2, y2, i, label] in track_bb_id:
                    x1, y1, x2, y2, i = list(
                        map(lambda x: int(float(x)), [x1, y1, x2, y2, i]))
                    cur_track[i] = [x1, y1, x2, y2, label]
                    # if the same object is found in last frame
                    if last_track is not None and i in last_track:
                        # and their positions are almost the same
                        if np.abs(
                                np.mean(cur_track[i][:4]) -
                                np.mean(last_track[i][:4])) < 1:
                            # add 1 to the count
                            static_count[i] = static_count.get(i, 0) + 1
                last_track = cur_track
                batch_track.append(cur_track)

            # check track IDs that are static in a batch
            static_id = []
            for i in cur_track:
                if static_count.get(i, 0) == len(frames):
                    static_id.append(i)

            if self.enable_bg_sub:
                self.cache_bg.put({
                    'ind': batch_ind,
                    'frames': frames,
                    'res_batch': nres_batch
                })
            dic = {
                'res_batch': nres_batch,
                'infer_time': t2 - t1,
                'decode': t,
                'static_id': static_id,
                'batch_track': batch_track,
                'mvs': mvs
            }

            self.communicator.send_infer_results(batch_ind, dic)

            if self.enable_bg_overlay:
                # overlay background
                self.overlay_background(batch_ind, frames[-1], static_id,
                                        cur_track)
                # add overlay res to infer res
                # for each frame in a batch
                for i in range(len(nres_batch)):
                    overlay_res_cur = []
                    overlay_ids = list(self.overlay_res.keys())

                    for j in overlay_ids:
                        label, x1, y1, x2, y2 = self.overlay_res[j]
                        # if overlay background is used and most of the
                        # image is foreground, which means the overlay
                        # object may have changed
                        if batch_ind > self.overlay_ind.get(
                                j, np.inf) and masks is not None:
                            if np.sum(masks[i][y1:y2, x1:x2] > 0) > 0.9 * (
                                    x2 - x1) * (y2 - y1):
                                # remove overlay res
                                self.overlay_ind.pop(j)
                                _, tx1, ty1, tx2, ty2 = self.overlay_res.pop(j)
                                # recover overlay bg from static background
                                self.background[
                                    ty1:ty2,
                                    tx1:tx2] = self.background_static[ty1:ty2,
                                                                      tx1:tx2]

                            # otherwise it means the overlay object is
                            # unchanged, just add it to infer res
                            else:
                                overlay_res_cur.append([label, x1, y1, x2, y2])
                                frames_infer[i][y1:y2,
                                                x1:x2] = self.background[y1:y2,
                                                                         x1:x2]

                    nres_batch[i] += overlay_res_cur
                    # nms filter
                    nres_batch[i] = nms_filter(nres_batch[i])

            if self.ground_truth_avail:
                self.perf.set_value(batch_ind, 'raw_size',
                                    self.gt.get_gt_size(batch_ind), 'byte',
                                    'sum')
                if self.task == 'OD':
                    precision, recall, f1_score, tp_box, fp_box, fn_box = \
                        self.gt.get_OD_acc_batch(batch_ind,
                                                 nres_batch)
                    if len(tp_box) != self.batch_size or len(
                            fp_box) != self.batch_size or len(
                                fn_box) != self.batch_size:

                        print(
                            f'dataset: {self.dataset_name}, ind: {batch_ind}, '
                            f'batch_size: {self.batch_size}, '
                            f'frames: {len(frames)}, '
                            f'nres_batch: {len(nres_batch)}, '
                            f'tp_box: {len(tp_box)}, fp_box: {len(fp_box)}, '
                            f'fn_box: {len(fn_box)}')
                        np.array(frames).tofile('frames.npy')
                        os._exit(0)
                    bboxes = [(tp_box[i], fp_box[i], fn_box[i])
                              for i in range(self.batch_size)]
                elif self.task == 'KD':
                    precision, recall, f1_score, tp_kps, fp_kps, fn_kps = \
                        self.gt.get_KD_acc_batch(batch_ind,
                                                 res_batch_KD)
                    res_batch_KD = [(tp_kps[i], fp_kps[i], fn_kps[i])
                                    for i in range(self.batch_size)]
                    bboxes = []  # do not show bboxes for KD
                self.perf.set_value(batch_ind,
                                    'precision',
                                    precision,
                                    '',
                                    'mean',
                                    digit=3)
                self.perf.set_value(batch_ind,
                                    'recall',
                                    recall,
                                    '',
                                    'mean',
                                    digit=3)
                self.perf.set_value(batch_ind,
                                    'f1_score',
                                    f1_score,
                                    '',
                                    'mean',
                                    digit=3)
            else:
                bboxes = nres_batch
            if self.show_res:
                self.cache_render.put({
                    'roi': frames,
                    'infer': frames_infer,
                    'rl': self.region_large,
                    'rs': self.region_small,
                    'ind': batch_ind,
                    'bboxes': bboxes,
                    'kps': res_batch_KD
                })

            self.perf.print(batch_ind)

    def background_reconstructor(self):
        for dic in iter(self.cache_bg.get, None):
            for i in range(len(dic['frames'])):
                self.bg_recon.update(dic['ind'], dic['frames'][i],
                                     dic['res_batch'][i])
                size = self.bg_recon.check_and_send_bg(dic['ind'],
                                                       self.communicator)
                # if bg image is sent
                if size > 0:
                    self.perf.set_value(dic['ind'], 'bg_size', size, 'byte',
                                        'sum')
                    self.background_static = self.bg_recon.last_sent_bg.copy()
                    self.background = self.bg_recon.last_sent_bg.copy()
                    self.overlay_res = {}

    def roi_updater(self, fps):
        ind = 0
        for dic in iter(self.cache_roi.get, None):
            # update ROI params
            batch_ind = dic['batch_ind']
            nres = dic['nres']
            self.roi_param.update_size_bb(nres)
            ind += 1
            if ind % (10 * fps) == 0:
                self.region_small, self.region_large, size = \
                    self.roi_param.send_det_roi_params(self.communicator, nres)
                # self.roi_param.send_roi_params(self.communicator)
                logger.info(
                    f'sending roi params on ind {batch_ind} ({size} B)')
                self.perf.set_value(dic['batch_ind'], 'roi_param_size', size,
                                    'byte', 'sum')

    def complement_roi_with_bg(self, roi, saveind):
        """fill roi region in black with background image, otherwise
        the CNN model will mis-detect some of the objects"""
        bg = self.bg_recon.get_bg()
        if bg is None:
            return roi, np.ones((self.height, self.width))

        mask_bg = (roi.astype('float').sum(axis=2) <
                   self.thresh_roi_black).astype('uint8')
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (self.macro_size, self.macro_size))
        mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_OPEN, kernel)
        mask_fg = 1 - mask_bg
        part_bg = cv2.bitwise_and(bg, bg, mask=mask_bg)
        part_fg = cv2.bitwise_and(roi, roi, mask=mask_fg)
        part_all = cv2.add(part_bg, part_fg)
        if self.save_intermediate and saveind % 10 == 0:
            cv2.imwrite(f'{self.inter_dir}/mask_bg_{saveind}.jpg',
                        (mask_bg * 255).astype('uint8'))
            cv2.imwrite(f'{self.inter_dir}/mask_fg_{saveind}.jpg',
                        (mask_fg * 255).astype('uint8'))
            cv2.imwrite(f'{self.inter_dir}/part_fg_{saveind}.jpg', part_fg)
            cv2.imwrite(f'{self.inter_dir}/part_bg_{saveind}.jpg', part_bg)
            cv2.imwrite(f'{self.inter_dir}/partall_{saveind}.jpg', part_all)
        return part_all, mask_fg

    def render(self):
        for dic in iter(self.cache_render.get, None):
            for i in range(self.batch_size):
                if self.save_intermediate and i == 0:
                    saveind = dic['ind'] * self.batch_size + i
                    cv2.imwrite(f'{self.inter_dir}/roi_{saveind}.png',
                                dic['roi'][i].copy())

                imgs_show = {}
                # show roi regions
                imgs_show[f"roi #{dic['ind']*self.batch_size+i}"] = dic['roi'][
                    i].copy()
                # show ROI params
                img_roi_param = dic['infer'][i].copy()
                if dic['rl'] is not None:
                    # yellow for large area, high QP
                    for x1, y1, x2, y2 in dic['rl']:
                        cv2.rectangle(img_roi_param, (x1, y1), (x2, y2),
                                      (0, 255, 255), 2)
                if dic['rs'] is not None:
                    # read for small area, low QP
                    for x1, y1, x2, y2 in dic['rs']:
                        cv2.rectangle(img_roi_param, (x1, y1), (x2, y2),
                                      (0, 0, 255), 2)
                if self.save_intermediate and i == 0:
                    cv2.imwrite(f'{self.inter_dir}/roiparam_{saveind}.png',
                                img_roi_param)

                imgs_show['ROI params'] = img_roi_param
                # show bg
                if self.save_intermediate and i == 0:
                    if self.background is not None:
                        cv2.imwrite(f'{self.inter_dir}/bg_{saveind}.png',
                                    self.background)
                        cv2.imwrite(
                            f'{self.inter_dir}/static_bg_{saveind}.png',
                            self.background_static)
                imgs_show['bg'] = self.background.copy(
                ) if self.background is not None else None

                # show infer results
                img_det = dic['infer'][i].copy()
                if self.task == 'OD':
                    img_det = add_boxes_to_img(dic['bboxes'][i], img_det,
                                               self.ground_truth_avail)
                elif self.task == 'KD':
                    img_det = add_kp_to_img(dic['kps'][i], img_det,
                                            self.ground_truth_avail)

                # save
                if self.save_intermediate and i == 0:
                    cv2.imwrite(f'{self.inter_dir}/det_{saveind}.png', img_det)
                imgs_show['det'] = img_det

                # send to show
                imgs_show_all = concat_image(imgs_show,
                                             size=(self.height, self.width))
                self.cache_show.put({'img': imgs_show_all})

                # save
                if self.save_intermediate and i == 0:
                    cv2.imwrite(f'{self.inter_dir}/all_{saveind}.png',
                                imgs_show_all)

    def show(self):
        t_last = time.time() - 1
        cv2.namedWindow('cloud', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('cloud', cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
        for dic in iter(self.cache_show.get, None):

            t_remain = 1 / self.fps - (time.time() - t_last) - 0.001
            if t_remain > 0:
                time.sleep(t_remain)
            # print(f"ind: {dic['batch_ind']:02d}, got: {dic['t_got']:.3f}"
            #       f", ready: {dic['t_ready']:.3f}, show: {time.time():.3f}")
            cv2.imshow('cloud', dic['img'])
            t_last = time.time()

            # q: quit, space: pause
            key = cv2.waitKey(1)
            if key == 32:
                cv2.waitKey(0)

    def check_finish(self):
        while True:
            if self.cache_show.empty(
            ) and self.communicator.get_finished_signal():
                logger.info('finished, closing service')
                self.perf.print_stat()
                res_path = str(self.res_dir / f'perf_{self.dataset_name}.csv')
                self.perf.save(res_path)
                os._exit(0)
            time.sleep(1)


def main(param):
    dataset_name = Path(param['video_path']).name

    cloud = Cloud(dataset_name=dataset_name,
                  task=param['task'],
                  model_type=param['model_type'],
                  hw_acc=param['encode_hardware_acceleration'],
                  resize=param['input_size'],
                  batch_size=param['batch_size'],
                  run_type=param['run_type'],
                  iou_th=param['iou_th'],
                  oks_th=param['oks_th'],
                  thresh_roi_black=param['thresh_roi_black'],
                  macro_size=param['macro_size'],
                  read_start=param['read_start'],
                  fps=param['fps'],
                  show_res=param['show_cloud'],
                  ground_truth_avail=param['ground_truth_avail'],
                  enable_bg_sub=param['enable_bg_sub'],
                  enable_bg_overlay=param['enable_bg_overlay'],
                  enable_roi_enc=param['enable_roi_enc'],
                  res_dir=param['res_dir'],
                  save_intermediate=param['save_intermediate'],
                  port=param['port'])

    tasks = []
    tasks.append(threading.Thread(target=cloud.infer, args=()))
    if param['enable_bg_sub']:
        tasks.append(
            threading.Thread(target=cloud.background_reconstructor, args=()))
    if param['show_cloud']:
        tasks.append(threading.Thread(target=cloud.render, args=()))
        tasks.append(threading.Thread(target=cloud.show, args=()))
    if param['enable_roi_enc']:
        tasks.append(
            threading.Thread(target=cloud.roi_updater, args=(param['fps'], )))
    tasks.append(threading.Thread(target=cloud.check_finish, args=()))

    for task in tasks:
        task.start()

    for task in tasks:
        task.join()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        param_path = 'param.yml'
    else:
        param_path = sys.argv[1]
    logger.info(f'loading params from {param_path}')

    with open(param_path) as f:
        param = yaml.load(f, Loader=yaml.FullLoader)

    main(param)
