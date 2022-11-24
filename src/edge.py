import os
import queue
import sys
import threading
import time
from pathlib import Path
import random

import cv2
import numpy as np
import yaml

from adaptive_weighting import AdaptiveWeighting
from comm import Emulation, Implementation
from offline_estimator import OfflineEstimator
from utils.misc import (concat_image, conv2d, dilate, get_logger, nms_filter,
                        resample, timing, wrapc, add_boxes_to_img)
from utils.perf import GroundTruth, PerfMeter, SystemResourceUsage
from utils.video_source import FromImgDir, FromRawVideo, FromCamera

logger = get_logger('edge', level='DEBUG')
random.seed(666)


class Edge:
    def __init__(self, dataset, task, run_type, batch_size, fps,
                 thresh_diff_smooth, ground_truth_avail, hw_acc, resize,
                 enable_bg_sub, enable_bg_overlay, enable_oe, oe_method,
                 enable_aw, aw_ratio, hname, port, macro_size, iou_th, oks_th,
                 num_batch, read_start, show_res, res_dir):

        assert task in ['OD', 'KD'], 'task must be one of Objection '
        'Detection (OD), or Keypoint Detection (KD)'
        logger.info('initializing edge side...')
        self.role = 'edge'
        self.dataset = dataset
        self.task = task
        if run_type == 'emulation':
            self.communicator = Emulation(role=self.role,
                                          read_start=read_start)
        elif run_type == 'implementation':
            self.communicator = Implementation(role=self.role,
                                               dataset_name=self.dataset.name,
                                               read_start=read_start,
                                               hname=hname,
                                               port=port)
        else:
            raise Exception(f'unkown run_type {run_type}')

        self.batch_size = batch_size
        self.fps = fps
        self.thresh_diff_smooth = thresh_diff_smooth
        self.ground_truth_avail = ground_truth_avail
        self.hw_acc = hw_acc
        self.enable_bg_sub = enable_bg_sub
        self.enable_bg_overlay = enable_bg_overlay
        self.enable_oe = enable_oe
        self.oe_method = oe_method
        self.enable_aw = enable_aw
        self.aw_ratio = aw_ratio
        self.num_batch = num_batch
        self.read_start = read_start
        self.show_res = show_res
        self.res_dir = Path(res_dir)
        self.res_dir.mkdir(parents=True, exist_ok=True)
        self.resize = resize

        self.background = None  # after overlay (for comparision)
        self.background_static = None  # without overlay (for recovering)
        self.img_bg_shift = None  # for visualization
        self.ind = self.read_start  # current frame index
        self.frames = None  # current frame batch
        self.roi = None  # roi bboxes for current frame
        self.res = None  # result for current frame
        self.height, self.width = None, None  # resolution of all the frames
        self.height_c, self.width_c = None, None  # resolution after conv
        self.cache_read = queue.Queue()  # cache to hold read data
        self.cache_roi = queue.Queue()  # cache to hold roi
        self.cache_infer = queue.Queue()  # cache to hold infer results
        self.cache_show = queue.Queue()  # cache to hold rendered images
        self.cache_render = queue.Queue()  # cache to render results
        self.macro_size = macro_size
        self.kernel_value = 1 / self.macro_size**2
        self.aw = AdaptiveWeighting(macro_size=macro_size)
        self.diff_smooth = None
        self.perf = PerfMeter()
        self.finish_ind = -1  # the last read ind
        if self.ground_truth_avail:
            self.gt = GroundTruth(task=self.task,
                                  vid_name=self.dataset.name,
                                  hw_acc=self.hw_acc,
                                  resize=resize,
                                  batch_size=batch_size,
                                  th_iou=iou_th,
                                  th_oks=oks_th,
                                  th_min_area=macro_size * macro_size)
        self.system_usage = SystemResourceUsage()
        self.oe = None  # offline estimator
        self.last_res = None  # last detection results, for offline estimator
        self.overlay_res = {}  # detection results of overlay objects
        self.overlay_ind = {}  # last batch index with overlay info updated
        self.sys_start = time.time()

        self.use_baseline = False
        self.drop = False
        self.transfer_raw = False

    def get_roi(self):
        """Compare with background and set bboxes roi"""
        # if there's no background image available
        if self.background is None or self.use_baseline:
            # send whole image to cloud for background reconstruction
            roi = self.frames
        else:
            # compare with background image to get roi
            t, roi = self.compare_with_background()
            self.perf.set_value(self.ind, 'get_roi', t, 's', 'mean')
        return roi

    @timing
    def compare_with_background(self):
        # per pixel diff (grayscale)
        t1 = time.time()
        frame_smooth = conv2d(image=self.frames,
                              kernel_value=self.kernel_value,
                              kernel_size=self.macro_size,
                              stride=self.macro_size,
                              channel_num=3)
        bg_smooth = conv2d(image=self.background,
                           kernel_value=self.kernel_value,
                           kernel_size=self.macro_size,
                           stride=self.macro_size,
                           channel_num=3)
        t2 = time.time()
        # per macro block (grayscale)
        self.diff_smooth = np.abs(frame_smooth - bg_smooth)
        t3 = time.time()
        if self.height_c is None:
            _, self.height_c, self.width_c = self.diff_smooth.shape
        if self.enable_aw:
            weight = self.aw.get_weight(frame_smooth)
        else:
            weight = None
        if weight is not None:
            if self.show_res:
                # for visualization only
                self.diff_smooth_before = self.diff_smooth.copy()
                self.weight = np.clip(weight * 63, 0, 255).astype('uint8')
            self.diff_smooth = self.diff_smooth * weight
        # threshold to get binary mask
        mask = np.zeros_like(self.diff_smooth)
        mask[self.diff_smooth > self.thresh_diff_smooth] = 1
        t4 = time.time()

        self.mask = dilate(mask, 3)
        t5 = time.time()
        self.mask = resample(self.mask,
                             (self.height, self.width)).astype('uint8')
        t6 = time.time()
        roi = self.frames.copy()
        t7 = time.time()
        roi = roi * np.stack((self.mask, ) * 3, axis=3)
        t8 = time.time()
        self.perf.set_value(self.ind, 'diff', t2 - t1, 's', 'mean')
        self.perf.set_value(self.ind, 'conv2d', t3 - t2, 's', 'mean')
        self.perf.set_value(self.ind, 'get_weight', t4 - t3, 's', 'mean')
        self.perf.set_value(self.ind, 'dilate', t5 - t4, 's', 'mean')
        self.perf.set_value(self.ind, 'resample', t6 - t5, 's', 'mean')
        self.perf.set_value(self.ind, 'copy', t7 - t6, 's', 'mean')
        self.perf.set_value(self.ind, 'mask', t8 - t7, 's', 'mean')
        return roi

    def run(self):
        logger.info('start...')

        t_last = time.time()
        logger.info('reading frames...')
        for ind, frames in self.dataset.read_batch(batch_size=self.batch_size,
                                                   start=self.read_start,
                                                   num_batch=self.num_batch):
            # print(f'reading frame {ind*self.batch_size} ~ \
            # {(ind+1)*self.batch_size}...')

            if ind < 459 * (1 - self.aw_ratio):
                self.use_baseline = True
            else:
                self.use_baseline = False

            if len(frames) == 0:
                logger.info('finished')
                break

            # read an image per 33ms
            t_remain = 1 / self.fps * self.batch_size - (time.time() - t_last)
            if t_remain > 0:
                time.sleep(t_remain)
            actual_fps = self.batch_size / (time.time() - t_last)
            self.perf.set_value(ind,
                                'afps',
                                actual_fps,
                                'fps',
                                'mean',
                                digit=1)
            t_last = time.time()

            # initialize height and width
            if self.height is None or self.width is None:
                self.height, self.width = frames[0].shape[:2]
                self.oe = OfflineEstimator(height=self.height,
                                           width=self.width,
                                           batch_size=self.batch_size)
            self.ind, self.frames = ind, frames

            if self.drop:
                roi = None
            else:
                if self.transfer_raw:
                    roi = self.frames
                else:
                    roi = self.get_roi()
            self.cache_roi.put([ind, roi])

            self.cache_read.put({
                'ind':
                ind,
                'frames':
                self.frames,
                'roi':
                roi,
                'diff_smooth':
                self.diff_smooth,
                'diff_smooth_before':
                self.diff_smooth_before
                if hasattr(self, 'diff_smooth_before') else None,
                'mask':
                self.mask if hasattr(self, 'mask') else None,
                'weight':
                self.weight if hasattr(self, 'weight') else None,
                't_start':
                t_last
            })

        self.finish_ind = ind

    def send_roi(self):
        roi_params_path = self.communicator.dir_misc / 'roi_params.txt'
        for ind, roi in iter(self.cache_roi.get, None):
            if self.use_baseline:
                roi_params_path = None
            t, size = self.communicator.send_roi(self.task, ind, roi,
                                                 self.hw_acc, roi_params_path)
            self.perf.set_value(ind, 'send_roi', t, 's', 'mean')
            if self.show_res:
                self.perf.set_value(ind, 'sent_size', size, 'byte', 'sum')

    def retrieve_infer_results(self):
        last_oe_update = True
        for ind, dic in self.communicator.get_infer_results():
            self.cache_infer.put({'ind': ind, 'dic': dic})

            # update weights
            if self.enable_aw and not self.use_baseline:
                overlay_res = list(self.overlay_res.values())
                self.aw.update(dic['res_batch'], overlay_res)

            # update offline estimator
            if self.enable_oe:
                if self.network_interrupt_mock(ind):
                    last_oe_update = False
                else:
                    if last_oe_update:
                        self.oe.update(dic['batch_track'])
                    last_oe_update = True

    def update_background(self):
        while True:
            background, size = self.communicator.get_bg()
            if background is not None:
                logger.info(f'updating background on ind {self.ind}'
                            f' ({size/1024:.2f} KB)')
                if self.show_res:
                    self.perf.set_value(self.ind, 'bg_size', size, 'byte',
                                        'sum')
                self.background = background
                self.background_static = background
                self.overlay_res = {}
            time.sleep(1)

    def update_roi_params(self):
        while True:
            size = self.communicator.get_roi_params()
            if size is not None:
                logger.info(
                    f'updating RoI params on ind {self.ind} ({size} B)')
            time.sleep(1)

    def overlay_background(self, ind, img, static_id, cur_track):
        if len(static_id) > 0 and self.background is not None:
            background = self.background.copy()

            for i in static_id:
                x1, y1, x2, y2, label = cur_track[str(i)]
                # overlay the background with objects
                background[y1:y2, x1:x2] = img[y1:y2, x1:x2]
                self.overlay_res[i] = [label, x1, y1, x2, y2]
                self.overlay_ind[i] = ind
            self.background = background

    def add_shift_info_to_bg(self):
        # show learned shift in offline estimator, only for visualization
        while True:
            if self.background is not None:
                if self.ind % 5 == 0:
                    img_bg = self.background.copy()
                    for i in range(0, len(img_bg), 10):
                        for j in range(0, len(img_bg[0]), 10):
                            nj = np.clip(j + int(self.oe.shift_x[i][j]), 0,
                                         self.width - 1)
                            ni = np.clip(i + int(self.oe.shift_y[i][j]), 0,
                                         self.height - 1)
                            cv2.arrowedLine(img_bg, (j, i), (nj, ni),
                                            (0, 0, 255), 1)
                    self.img_bg_shift = img_bg
            time.sleep(1)

    def exit(self):
        self.perf.print_stat()
        res_path = str(self.res_dir / f'perf_{self.dataset.name}.csv')
        self.perf.save(res_path)
        self.communicator.send_finished_signal()
        self.system_usage.exit()
        self.sys_end = time.time()
        print(f'Total time: {self.sys_end-self.sys_start} s')
        os._exit(0)

    def estimate_optical_flow(self, prev_img, cur_img_batch, prev_det,
                              plot_shift):
        bbox_batch = []
        rows, cols, _ = cur_img_batch[0].shape

        for i in range(len(cur_img_batch)):
            prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
            cur_gray = cv2.cvtColor(cur_img_batch[i], cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5,
                                                3, 15, 3, 5, 1.2, 0)

            # plot shift
            if plot_shift:
                img_bg = self.background.copy()
                for oi in range(0, len(img_bg), 10):
                    for oj in range(0, len(img_bg[0]), 10):
                        nj = np.clip(oj + int(flow[oi, oj, 0]), 0,
                                     self.width - 1)
                        ni = np.clip(oi + int(flow[oi, oj, 1]), 0,
                                     self.height - 1)
                        cv2.arrowedLine(img_bg, (oj, oi), (nj, ni),
                                        (0, 0, 255), 1)
                self.img_bg_shift = img_bg

            bbox = []
            for (label, x1, y1, x2, y2) in prev_det:
                # calculate optical flow shift
                if (x2 - x1) * (y2 - y1) == 0:
                    continue

                x_shift = int(np.mean(flow[y1:y2, x1:x2, 0]))
                y_shift = int(np.mean(flow[y1:y2, x1:x2, 1]))

                # calculate new bbox
                x1 = min(cols - 1, max(0, x1 + x_shift))
                x2 = min(cols - 1, max(0, x2 + x_shift))
                y1 = min(rows - 1, max(0, y1 + y_shift))
                y2 = min(rows - 1, max(0, y2 + y_shift))

                bbox.append([label, x1, y1, x2, y2])

            prev_det = bbox
            prev_img = cur_img_batch[i]
            bbox_batch.append(bbox)
        return bbox_batch

    def estimate_mv(self, prev_det, mvs):
        bbox_batch = []
        rows, cols = self.resize

        for i in range(self.batch_size):
            mv = mvs[i]
            if len(mv) == 0:
                bbox_batch.append([])
                continue

            flow = np.zeros((rows, cols, 2))
            for each in mv:
                block_width = each[1]
                block_height = each[2]
                x = each[3]
                y = each[4]
                shift_x = each[5] - each[3]
                shift_y = each[6] - each[4]
                flow[y - block_height // 2:y + block_height // 2,
                     x - block_width // 2:x + block_width, 0] = shift_x
                flow[y - block_height // 2:y + block_height // 2,
                     x - block_width // 2:x + block_width, 1] = shift_y

            bbox = []
            for (label, x1, y1, x2, y2) in prev_det:
                # calculate optical flow shift
                if (x2 - x1) * (y2 - y1) == 0:
                    continue

                x_shift = int(np.mean(flow[y1:y2, x1:x2, 0]))
                y_shift = int(np.mean(flow[y1:y2, x1:x2, 1]))

                # calculate new bbox
                x1 = min(cols - 1, max(0, x1 + x_shift))
                x2 = min(cols - 1, max(0, x2 + x_shift))
                y1 = min(rows - 1, max(0, y1 + y_shift))
                y2 = min(rows - 1, max(0, y2 + y_shift))

                bbox.append([label, x1, y1, x2, y2])

            prev_det = bbox
            bbox_batch.append(bbox)
        return bbox_batch

    def network_interrupt_mock(self, ind):
        if ind >= 100 and ind % 5 == 0:
            return True
        else:
            return False

    def postprocess(self):

        while True:
            c_read = self.cache_read.get()
            c_infer = self.cache_infer.get()
            if time.time() - c_read['t_start'] > 3:
                self.drop = True
            else:
                self.drop = False

            # if time.time() - c_read['t_start'] < 1.5:
            #     self.transfer_raw = True
            # else:
            #     self.transfer_raw = False

            if c_infer['dic']['infer_time'] == -1:
                self.perf.set_value(c_read['ind'], wrapc('total', 'r'),
                                    time.time() - c_read['t_start'], 's',
                                    'mean')
                cpu, mem, gpu = self.system_usage.get_stats()
                self.perf.set_value(c_read['ind'], 'CPU', cpu, '%', 'mean')
                self.perf.set_value(c_read['ind'], 'Memory', mem, '%', 'mean')
                self.perf.set_value(c_read['ind'], 'GPU', gpu, '%', 'mean')
                self.perf.print(c_read['ind'])
                if c_read['ind'] == self.finish_ind:
                    self.exit()
                continue

            assert c_read['ind'] == c_infer['ind'], (
                f"incompatible ind found in read (ind: {c_read['ind']})"
                f" and infer (ind: {c_infer['ind']})")
            t_post_process_start = time.time()

            # estimate with probability
            if self.enable_oe and self.network_interrupt_mock(
                    c_read['ind']) and self.last_res is not None:
                estimated = True
                method = self.oe_method  # 'worst'
                assert method in ['oe', 'of', 'worst', 'best', 'mv', 'mv_oe']

                t1 = time.time()
                # estimate with mv + oe
                if method == 'mv_oe':
                    c_infer['dic']['res_batch'] = self.oe.estimate(
                        self.last_res, self.batch_size,
                        self.estimate_mv(self.last_res, c_infer['dic']['mvs']))
                elif method == 'oe':
                    c_infer['dic']['res_batch'] = self.oe.estimate_old(
                        self.last_res, self.batch_size)
                # estimate with optical flow
                elif method == 'of':
                    c_infer['dic']['res_batch'] = self.estimate_optical_flow(
                        self.last_img, c_read['frames'], self.last_res, False)
                # no estimation to calculate gt
                elif method == 'worst':
                    c_infer['dic']['res_batch'] = [
                        [] for _ in range(self.batch_size)
                    ]
                # using gt as estimation
                elif method == 'best':
                    pass
                elif method == 'mv':
                    c_infer['dic']['res_batch'] = self.estimate_mv(
                        self.last_res, c_infer['dic']['mvs'])
                else:
                    raise Exception(f'unkown oe method {method}')
                t2 = time.time()
                estimate_time = t2 - t1

            else:
                estimated = False
            self.last_res = c_infer['dic']['res_batch'][-1]
            self.last_img = c_read['frames'][-1]

            if self.enable_bg_overlay and not self.use_baseline:
                if len(c_infer['dic']['batch_track']) == 0:
                    cur_track = {}
                else:
                    cur_track = c_infer['dic']['batch_track'][-1]
                # overlay background
                self.overlay_background(c_read['ind'], c_read['frames'][-1],
                                        c_infer['dic']['static_id'], cur_track)

                # add overlay res to infer res
                # for each frame in a batch
                for i in range(len(c_infer['dic']['res_batch'])):
                    overlay_res_cur = []
                    overlay_ids = list(self.overlay_res.keys())

                    for j in overlay_ids:
                        label, x1, y1, x2, y2 = self.overlay_res[j]
                        # if overlay background is used and most of the
                        # image is foreground, which means the overlay
                        # object may have changed
                        if c_read['ind'] > self.overlay_ind.get(
                                j, np.inf) and c_read['mask'] is not None:
                            if np.sum(c_read['mask'][i][y1:y2, x1:x2] > 0
                                      ) > 0.9 * (x2 - x1) * (y2 - y1):
                                # remove overlay res
                                self.overlay_ind.pop(j)
                                _, tx1, ty1, tx2, ty2 = self.overlay_res.pop(j)
                                # recover overlay bg from static background
                                self.background[
                                    ty1:ty2,
                                    tx1:tx2] = self.background_static[ty1:ty2,
                                                                      tx1:tx2]

                            # otherwise it means the overlay object is unchanged,
                            # just add it to infer res
                            else:
                                overlay_res_cur.append([label, x1, y1, x2, y2])

                    c_infer['dic']['res_batch'][i] += overlay_res_cur
                    # nms filter
                    c_infer['dic']['res_batch'][i] = nms_filter(
                        c_infer['dic']['res_batch'][i])

            t_start = c_read['t_start']
            t_end = time.time()
            self.perf.set_value(c_read['ind'], 'infer',
                                c_infer['dic']['infer_time'], 's', 'mean')
            self.perf.set_value(c_read['ind'], 'decode',
                                c_infer['dic']['decode'], 's', 'mean')
            self.perf.set_value(c_read['ind'], 'postprocess',
                                t_end - t_post_process_start, 's', 'mean')
            self.perf.set_value(c_read['ind'], wrapc('total', 'r'),
                                t_end - t_start, 's', 'mean')

            # check cpu, mem and gpu usage
            cpu, mem, gpu = self.system_usage.get_stats()
            self.perf.set_value(c_read['ind'], 'CPU', cpu, '%', 'mean')
            self.perf.set_value(c_read['ind'], 'Memory', mem, '%', 'mean')
            self.perf.set_value(c_read['ind'], 'GPU', gpu, '%', 'mean')

            if estimated:
                _, _, f1_score, tp_box, fp_box, fn_box = self.gt.get_OD_acc_batch(
                    c_read['ind'], c_infer['dic']['res_batch'])
                self.perf.set_value(c_read['ind'],
                                    wrapc('estimate_f1_score', 'y'),
                                    f1_score,
                                    '',
                                    'mean',
                                    digit=3)
                self.perf.set_value(c_read['ind'],
                                    wrapc('estimate_time', 'y'),
                                    estimate_time,
                                    's',
                                    'mean',
                                    digit=2)

            if self.show_res:
                if self.ground_truth_avail:
                    precision, recall, f1_score, tp_box, fp_box, fn_box = \
                        self.gt.get_OD_acc_batch(c_read['ind'],
                                                 c_infer['dic']['res_batch'])
                    bboxes = [(tp_box[i], fp_box[i], fn_box[i])
                              for i in range(self.batch_size)]
                    self.perf.set_value(c_read['ind'],
                                        'precision',
                                        precision,
                                        '',
                                        'mean',
                                        digit=3)
                    self.perf.set_value(c_read['ind'],
                                        'recall',
                                        recall,
                                        '',
                                        'mean',
                                        digit=3)
                    self.perf.set_value(c_read['ind'],
                                        'f1_score',
                                        f1_score,
                                        '',
                                        'mean',
                                        digit=3)
                    self.perf.set_value(c_read['ind'], 'raw_size',
                                        self.gt.get_gt_size(c_read['ind']),
                                        'byte', 'sum')
                else:
                    bboxes = c_infer['dic']['res_batch']
                self.cache_render.put({
                    'c_read': c_read,
                    'bboxes': bboxes,
                    'estimated': estimated
                })
                self.perf.print(c_read['ind'])
            else:
                self.perf.print(c_read['ind'])
                if c_read['ind'] == self.finish_ind:
                    self.exit()

    def render(self):
        # bgr colors
        frame_ind = self.ind * self.batch_size

        for data in iter(self.cache_render.get, None):
            c_read = data['c_read']
            bboxes = data['bboxes']
            estimated = data['estimated']
            for i in range(len(c_read['frames'])):
                imgs_show = {}  # e.g.: {'backgroud': bg}

                frame_ind += 1
                # add bounding boxes
                img = add_boxes_to_img(bboxes[i], c_read['frames'][i].copy(),
                                       self.ground_truth_avail)
                imgs_show[
                    f'#{frame_ind} {"estimated" if estimated else ""}'] = img
                imgs_show['diff_smooth'] = c_read['diff_smooth'][i] if c_read[
                    'diff_smooth'] is not None else None
                imgs_show['diff_smooth_before'] = c_read['diff_smooth_before'][
                    i] if c_read['diff_smooth_before'] is not None else None
                imgs_show['background'] = self.img_bg_shift
                imgs_show['roi'] = c_read['roi'][i]
                imgs_show['weight'] = c_read['weight'] if c_read[
                    'weight'] is not None else None
                # imgs_show['mask'] = c_read['mask'][i] * 255 if c_read[
                #    'mask'] is not None else None

                # show with 30 fps
                imgs_show = concat_image(imgs_show,
                                         size=(self.height, self.width))
                if estimated:
                    Path('./tmp/estimate').mkdir(parents=True, exist_ok=True)
                    cv2.imwrite('./tmp/estimate/%d.jpg' % frame_ind, imgs_show)
                self.cache_show.put({
                    'start': c_read['t_start'],
                    'ready': time.time(),
                    'ind': c_read['ind'],
                    'img': imgs_show
                })

    def show(self):
        # t_last = time.time() - 1
        # display fullscreen
        cv2.namedWindow('edge', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('edge', cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
        for imgs_show in iter(self.cache_show.get, None):

            cv2.imshow('edge', imgs_show['img'])
            # t_last = time.time()
            # logger.info(
            #     f"ind: {imgs_show['ind']:03d}, read_finish: "
            #     f"{imgs_show['start']:.3f}, "
            #     f"ready: {imgs_show['ready']:.3f}, show: {t_last:.3f}, "
            #     f"ready_delay: {imgs_show['ready']-imgs_show['start']:.3f},"
            #     f" show_delay: {t_last-imgs_show['start']:.3f}")

            # q: quit, space: pause
            key = cv2.waitKey(20)
            if key == ord('q'):
                cv2.destroyAllWindows()
                self.exit()
            elif key == 32:
                cv2.waitKey(0)

            # exit after showing all results
            if imgs_show['ind'] == self.finish_ind:
                self.exit()


def main(param):
    video_path = param['video_path']
    if video_path == 'camera':
        dataset = FromCamera(resize=param['input_size'], fps=param['fps'])
    else:
        if video_path.endswith('.mp4'):
            dataset = FromRawVideo(src_file=video_path,
                                   resize=param['input_size'])
        else:
            dataset = FromImgDir(dst_dir=video_path,
                                 resize=param['input_size'],
                                 filetype=param['img_type'])
    logger.info(f'perform video analytics on {dataset.name}...')

    edge = Edge(dataset=dataset,
                task=param['task'],
                run_type=param['run_type'],
                batch_size=param['batch_size'],
                fps=param['fps'],
                thresh_diff_smooth=param['thresh_diff_smooth'],
                ground_truth_avail=param['ground_truth_avail'],
                hw_acc=param['encode_hardware_acceleration'],
                resize=param['input_size'],
                enable_bg_sub=param['enable_bg_sub'],
                enable_bg_overlay=param['enable_bg_overlay'],
                enable_oe=param['enable_oe'],
                oe_method=param['oe_method'],
                enable_aw=param['enable_aw'],
                aw_ratio=param['aw_ratio'],
                hname=param['hname'],
                port=param['port'],
                macro_size=param['macro_size'],
                iou_th=param['iou_th'],
                oks_th=param['oks_th'],
                num_batch=param['num_batch'],
                read_start=param['read_start'],
                show_res=param['show_edge'],
                res_dir=param['res_dir'])

    tasks = []
    tasks.append(threading.Thread(target=edge.run, args=()))
    tasks.append(threading.Thread(target=edge.send_roi, args=()))
    tasks.append(threading.Thread(target=edge.retrieve_infer_results, args=()))
    if param['enable_bg_sub']:
        tasks.append(threading.Thread(target=edge.update_background, args=()))
    tasks.append(threading.Thread(target=edge.add_shift_info_to_bg, args=()))
    tasks.append(threading.Thread(target=edge.postprocess, args=()))
    if param['enable_roi_enc']:
        tasks.append(threading.Thread(target=edge.update_roi_params, args=()))
    if param['show_edge']:
        tasks.append(threading.Thread(target=edge.render, args=()))
        tasks.append(threading.Thread(target=edge.show, args=()))
    tasks.append(threading.Thread(target=edge.system_usage.run, args=()))

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
