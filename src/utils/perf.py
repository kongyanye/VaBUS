import importlib.util
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

cmd = 'systemctl is-active --quiet jetson_stats.service'
# use jtop to get stats for jetson (only when jtop is running)
if importlib.util.find_spec('jtop') and os.system(cmd) == 0:
    from jtop import jtop
    use_jtop = True
# else use psutil and GPUtil to get stats
else:
    import GPUtil
    import psutil
    use_jtop = False


class PerfMeter():
    def __init__(self):
        """
            Args:
                perf (dict of dict): {ind: {step_name: time_elapsed}}
        """
        self.perf = {}
        self.unit = {}
        self.stats = {}
        self.values = {}
        self.digit = {}
        # automatically transform these units
        self.trans = {
            's': {
                'unit': 'ms',
                'multi': 1000
            },
            'byte': {
                'unit': 'KB',
                'multi': 1 / 1024
            }
        }

    def set_value(self, ind, k, v, unit, stat, digit=2):
        """
        Args:
            digit: number of decimal points to show
        """
        assert stat in ['mean', 'sum'], 'only mean and sum are'
        f'supported yet, got {stat}'

        if unit in self.trans:
            target_unit = self.trans[unit]['unit']
            target_v = v * self.trans[unit]['multi']
        else:
            target_unit = unit
            target_v = v

        if ind not in self.perf:
            self.perf[ind] = {}
        self.perf[ind][k] = target_v
        if k not in self.unit:
            self.unit[k] = target_unit
        if k not in self.stats:
            self.stats[k] = stat
        if k not in self.digit:
            self.digit[k] = digit

        self.values[k] = self.values.get(k, []) + [target_v]

    def print(self, ind):
        if ind not in self.perf:
            raise Exception(f'ind {ind} not found in self.perf')
        dic = self.perf[ind]
        line = [f'batch ind: {ind}']
        for k, v in dic.items():
            line.append(f'{k}: {v:.{self.digit[k]}f} {self.unit[k]}'.rstrip())
        print(', '.join(line))

    def print_stat(self):
        print('\nStats:\n' + '*' * 30)
        lines = []
        sort_t = {}
        for k, v in self.values.items():
            if self.stats[k] == 'mean':
                v_val = np.mean(v)
                v_std = np.std(v)
                v_str = (f'{v_val:.{self.digit[k]}f} ± '
                         f'2*{v_std:.{self.digit[k]}f}')
            elif self.stats[k] == 'sum':
                v_val = np.sum(v)
                v_str = f'{v_val:.{self.digit[k]}f}'
            else:
                raise Exception(f'Unknown stats {self.stats[k]}')
            line = f'{self.stats[k]} {k}: {v_str} {self.unit[k]}'.rstrip()
            if self.stats[k] == 'mean':
                min_val = np.min(v)
                max_val = np.max(v)
                line += (f', range: {min_val:.{self.digit[k]}f} ~ '
                         f'{max_val:.{self.digit[k]}f}')
            lines.append(line)
            sort_t[line] = (self.unit[k], v_val)
        # sort first by unit, then by value
        lines = sorted(lines, key=lambda x: sort_t[x])
        for i, line in enumerate(lines):
            if i > 0 and sort_t[line][0] != sort_t[lines[i - 1]][0]:
                print('-' * 30)
            print(line)
        print('*' * 30 + '\n')

    def save(self, fname):
        print(f'saving to {fname}')
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame.from_dict(self.perf, orient='index')
        cols = [
            each[8:-4] if each.endswith('\033[0m') else each
            for each in df.columns
        ]
        df.columns = cols
        df.to_csv(fname, index=False)


class SystemResourceUsage:
    """Monitor system resource usage"""
    def __init__(self):
        self.reset()
        self.lock = threading.Lock()
        self.use_jtop = use_jtop
        # if on jetson, use jtop to get stats
        if self.use_jtop:
            self.get_current_stats = self.get_current_stats_npu
            self.jtop = jtop()
            self.jtop.start()
        # else use psutil and GPUtil to get stats
        else:
            self.get_current_stats = self.get_current_stats_gpu
            self.use_jtop = False

    def reset(self):
        """Reset usage to zero"""
        self.cpu_usage = 0
        self.mem_usage = 0
        self.gpu_usage = 0
        self.num = 0

    def exit(self):
        if self.use_jtop:
            self.jtop.close()

    def get_current_stats_gpu(self):
        """Get current cpu, memory and gpu usage stats for GPU

        Returns:
            percentage of usage
        """
        # get average cpu usage percent
        cpu_usage = psutil.cpu_percent()
        # get memory usage percent
        mem = psutil.virtual_memory()
        mem_usage = mem.percent
        # get average GPU usage percent
        gpus = GPUtil.getGPUs()
        if len(gpus) > 0:
            gpu_usage = np.mean([gpu.memoryUtil * 100 for gpu in gpus])
        else:
            gpu_usage = 0
        return cpu_usage, mem_usage, gpu_usage

    def get_current_stats_npu(self):
        """Get current cpu, memory and gpu usage stats for NPU (jetson)

        Returns:
            percentage of usage
        """
        # get average cpu usage percent
        cpu_usage = np.mean(
            [self.jtop.cpu[each]['val'] for each in self.jtop.cpu])
        # get memory usage percent
        mem_usage = self.jtop.ram['use'] / self.jtop.ram['tot'] * 100
        # get average GPU usage percent
        gpu_usage = self.jtop.gpu['val']
        return cpu_usage, mem_usage, gpu_usage

    def get_stats(self):
        """Get cpu, memory and gpu usage stats over a period of time"""
        self.lock.acquire()
        if self.num == 0:
            self.lock.release()
            return 0, 0, 0
        else:
            cpu_mean = self.cpu_usage / self.num
            mem_mean = self.mem_usage / self.num
            gpu_mean = self.gpu_usage / self.num
            self.reset()
            self.lock.release()
            return cpu_mean, mem_mean, gpu_mean

    def run(self, interval=1):
        """Run to monitor resource usage

        Args:
            interval (int): log usage every 'interval' seconds
        """
        last_t = time.time()
        while True:
            t_remain = interval - (time.time() - last_t)
            if t_remain > 0:
                time.sleep(t_remain)

            cpu, mem, gpu = self.get_current_stats()
            self.lock.acquire()
            self.cpu_usage += cpu
            self.mem_usage += mem
            self.gpu_usage += gpu
            self.num += 1
            self.lock.release()
            last_t = time.time()


def iou(bbox1, bbox2):
    """
    Args:
        bbox1: (x1, y1, x2, y2), where (x1, y1) is top-left corner
            (x2, y2) is bottom-right corner
        bbox2: sames as bbox1

    Returns:
        iou of bbox1 and bbox2
    """
    ax1, ay1, ax2, ay2 = bbox1
    bx1, by1, bx2, by2 = bbox2
    x1_max = max(ax1, bx1)
    x2_min = min(ax2, bx2)
    y1_max = max(ay1, by1)
    y2_min = min(ay2, by2)
    if x1_max > x2_min or y1_max > y2_min:
        return 0
    else:
        overlap = (x2_min - x1_max) * (y2_min - y1_max)
        total = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1)
        return overlap / (total - overlap)


COCO_KEYPOINTS = [
    'nose',  # 1
    'left_eye',  # 2
    'right_eye',  # 3
    'left_ear',  # 4
    'right_ear',  # 5
    'left_shoulder',  # 6
    'right_shoulder',  # 7
    'left_elbow',  # 8
    'right_elbow',  # 9
    'left_wrist',  # 10
    'right_wrist',  # 11
    'left_hip',  # 12
    'right_hip',  # 13
    'left_knee',  # 14
    'right_knee',  # 15
    'left_ankle',  # 16
    'right_ankle',  # 17
]

COCO_PERSON_SIGMAS = [
    0.026,  # nose
    0.025,  # eyes
    0.025,  # eyes
    0.035,  # ears
    0.035,  # ears
    0.079,  # shoulders
    0.079,  # shoulders
    0.072,  # elbows
    0.072,  # elbows
    0.062,  # wrists
    0.062,  # wrists
    0.107,  # hips
    0.107,  # hips
    0.087,  # knees
    0.087,  # knees
    0.089,  # ankles
    0.089,  # ankles
]


def oks(dic1, dic2):
    '''Calculate Object Keypoint Similarity between
    two humans

    Args:
        dic1, dic2: dict of {k: v}, k is keypoint ID, v is
            a list of [x, y, conf]

    '''
    if len(dic1.keys()) == 0 and len(dic2.keys()) == 0:
        return 1

    # dic1 = {}
    # for (a, b, c, d) in res1.decoding_order:
    #     dic1[a] = c
    #     dic1[b] = d
    # dic2 = {}
    # for (a, b, c, d) in res2.decoding_order:
    #     dic2[a] = c
    #     dic2[b] = d

    common_keys = set(dic1.keys()) & set(dic2.keys())
    if len(common_keys) == 0:
        return 0

    kp1, kp2 = [], []
    sigmas = []
    for key in common_keys:
        kp1.append(dic1[key])
        kp2.append(dic2[key])
        sigmas.append(COCO_PERSON_SIGMAS[key])
    kp1, kp2 = np.array(kp1), np.array(kp2)
    sigmas = np.array(sigmas)

    xmax = np.max(np.vstack((kp1[:, 0], kp2[:, 0])))
    xmin = np.min(np.vstack((kp1[:, 0], kp2[:, 0])))
    ymax = np.max(np.vstack((kp1[:, 1], kp2[:, 1])))
    ymin = np.min(np.vstack((kp1[:, 1], kp2[:, 1])))
    s = (xmax - xmin) * (ymax - ymin)
    d = np.sum((kp1[:, :2] - kp2[:, :2])**2, axis=1)
    oks = np.mean(np.exp(-d**2 / (2 * s * sigmas**2)))
    return oks


class GroundTruth:
    def __init__(self,
                 task,
                 vid_name,
                 hw_acc,
                 resize,
                 batch_size,
                 th_iou=0.5,
                 th_oks=0.5,
                 th_min_area=256,
                 ground_truth_dir='../dataset/ground_truth'):
        """
        Args:
            vid_name (str): dataset name to check performance
            hw_acc (bool): whether hardware acceleration of video encoding
                is used or not. If no, use iio encoded size. Otherwise,
                use hw encoded size.
            batch_size (int): batch size in returned results, e.g., 15
            th_iou (float): threshold to consider a bounding box is
                correctly detected, e.g., 0.5
            th_min_area (int): minimum detection area size to consider
                in ground truth and returned results, unit is number
                of pixels, e.g., 256
        """
        self.task = task
        assert self.task in ['OD', 'KD'], ('expected task in [OD, KD], got'
                                           f'{self.task}')
        self.vid_name = vid_name
        self.hw_acc = hw_acc
        self.resize = resize
        self.batch_size = batch_size
        # iou threshold to decide whether a detection is right
        self.th_iou = th_iou
        self.th_oks = th_oks
        self.th_min_area = th_min_area
        self.ground_truth_dir = ground_truth_dir

        if resize is None:
            resize_type = 'rawsize'
        elif isinstance(resize, list):
            resize_type = f"resize_{'_'.join([str(each) for each in resize])}"
        else:
            resize_type = f'resize_{resize}'
        encode_type = 'hw' if self.hw_acc else 'iio'
        self.gt_encode_dir = Path(self.ground_truth_dir) / Path(
            f"encode_{resize_type}_batch_{batch_size}_{encode_type}")

        self.gt_det_dir = Path(
            self.ground_truth_dir) / Path(f'{self.task}_{resize_type}')

        # check dir existence
        if not self.gt_det_dir.is_dir():
            raise Exception(f'dir {self.gt_det_dir} does not exist')
        if not self.gt_encode_dir.is_dir():
            raise Exception(f'dir {self.gt_encode_dir} does not exist')

        # load gt results
        self.load_gt()

    def load_gt(self):
        self.gt_det = pd.read_csv(self.gt_det_dir / f'{self.vid_name}.csv')
        if self.task == 'OD':
            # filter object detection results
            self.gt_det = self.gt_det[(self.gt_det['x2'] - self.gt_det['x1']) *
                                      (self.gt_det['y2'] - self.gt_det['y1'])
                                      >= self.th_min_area].reset_index(
                                          drop=True)

            for each in ['x1', 'x2', 'y1', 'y2']:
                self.gt_det[each] = self.gt_det[each].astype('int')
        elif self.task == 'KD':
            for each in ['x', 'y']:
                self.gt_det[each] = self.gt_det[each].astype('int')
            self.gt_det['x'] = self.gt_det['x'] / 641 * self.resize[1]
            self.gt_det['y'] = self.gt_det['y'] / 369 * self.resize[0]

        self.gt_encode_df = pd.read_csv(self.gt_encode_dir /
                                        f'{self.vid_name}.csv')

    def get_gt_OD(self, ind):
        """get ground truth Object Detection result for image index ind

        Args:
            ind (int): image index, one res per image
        """
        gt = self.gt_det[self.gt_det['ind'] == ind]
        return gt

    def get_OD_acc(self, ind, res):
        # filter res
        nres = []
        for label, x1, y1, x2, y2 in res:
            if (x2 - x1) * (y2 - y1) >= self.th_min_area:
                nres.append([label, x1, y1, x2, y2])
        res = nres

        gt = self.get_gt_OD(ind)
        tp, fp, fn = 0, 0, 0
        tp_box, fp_box, fn_box = [], [], []
        for bbox1 in res:
            found = False
            for bbox2 in gt[['label', 'x1', 'y1', 'x2', 'y2']].values:
                # if label unmatched
                if bbox1[0] != bbox2[0]:
                    continue
                if iou(bbox1[1:], bbox2[1:]) >= self.th_iou:
                    found = True
                    break
            if found:
                tp += 1
                tp_box.append(bbox1)
            else:
                fp += 1
                fp_box.append(bbox1)

        for bbox2 in gt[['label', 'x1', 'y1', 'x2', 'y2']].values:
            found = False
            for bbox1 in res:
                # if label unmatched
                if bbox1[0] != bbox2[0]:
                    continue
                if iou(bbox1[1:], bbox2[1:]) >= self.th_iou:
                    found = True
                    break
            if not found:
                fn += 1
                fn_box.append(bbox2)

        precision = tp / (tp + fp) if (tp + fp) != 0 else 1
        recall = tp / (tp + fn) if (tp + fn) != 0 else 1
        f1_score = 2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp +
                                                       fn) != 0 else 1
        return precision, recall, f1_score, tp_box, fp_box, fn_box

    def get_OD_acc_batch(self, batch_ind, res):
        """calculate precision, recall and f1 score for image batch ind,
        the metrics are over number of images per batch

        Args:
            batch_ind (int): batch image index (NOT) per image index
            res (list of list): detection result for batch images
        """

        ind_start = batch_ind * self.batch_size
        # always return the same number of results as in res,
        # some videos has no objectes detected in ending frames,
        # so the entry in ground truth is missing
        ind_end = ind_start + len(res)
        ind_list = list(range(ind_start, ind_end))

        precision_all = []
        recall_all = []
        f1_score_all = []
        tp_box_all = []
        fp_box_all = []
        fn_box_all = []
        for i, ind in enumerate(ind_list):
            precision, recall, f1_score, tp_box, fp_box, fn_box = \
                self.get_OD_acc(ind, res[i])
            precision_all.append(precision)
            recall_all.append(recall)
            f1_score_all.append(f1_score)
            tp_box_all.append(tp_box)
            fp_box_all.append(fp_box)
            fn_box_all.append(fn_box)

        return np.mean(precision_all), np.mean(recall_all), np.mean(
            f1_score_all), tp_box_all, fp_box_all, fn_box_all

    def get_gt_size(self, batch_ind):
        """get encoded raw video size for batch image index ind

        Args:
            batch_ind (int): batch image index (NOT per image index)
        """
        size_values = self.gt_encode_df.loc[self.gt_encode_df['ind'] ==
                                            batch_ind, 'size'].values
        if len(size_values) == 0:
            raise Exception(
                f"batch ind {batch_ind} not found in {str(self.gt_encode_dir)}"
                f"/{self.vid_name}.csv. This may be due to only part of "
                "encoding results are generated. Consider regenerate encode "
                "size ground truth.")
        return size_values[0]

    def get_compress_ratio(self, batch_ind, size):
        size_raw = self.get_gt_size(batch_ind)
        return size / size_raw

    def get_gt_KD(self, ind):
        """get ground truth Human Keypoint Detection result for image
        index ind

        Args:
            ind (int): image index, one res per image
        """
        gt = self.gt_det[self.gt_det['ind'] == ind]
        dics = []
        human_inds = gt['human_ind'].unique()
        for human_ind in human_inds:
            dic = {}
            gt_human = gt[gt['human_ind'] == human_ind]
            for i in range(len(gt_human)):
                kp_ind, x, y, conf = gt_human.iloc[
                    i, [2, 3, 4, 5]].values
                dic[int(kp_ind)] = [x, y, conf]
            dics.append(dic)
        return dics

    def preprocess_res_KD(self, res):
        dics = []
        for each in res:
            dic = {}
            for (a, b, c, d) in each.decoding_order:
                dic[a] = list(c)
                dic[b] = list(d)
            for k in dic:
                dic[k][0] = dic[k][0] / 641 * self.resize[1]
                dic[k][1] = dic[k][1] / 369 * self.resize[0]
            dics.append(dic)
        return dics

    def get_KD_acc(self, ind, res):

        gt = self.get_gt_KD(ind)
        res = self.preprocess_res_KD(res)
        tp, fp, fn = 0, 0, 0
        tp_box, fp_box, fn_box = [], [], []
        for dic1 in res:
            found = False
            for dic2 in gt:
                if oks(dic1, dic2) >= self.th_oks:
                    found = True
                    break
            if found:
                tp += 1
                tp_box.append(dic1)
            else:
                fp += 1
                fp_box.append(dic1)

        for dic2 in gt:
            found = False
            for dic1 in res:
                if oks(dic1, dic2) >= self.th_oks:
                    found = True
                    break
            if not found:
                fn += 1
                fn_box.append(dic2)

        precision = tp / (tp + fp) if (tp + fp) != 0 else 1
        recall = tp / (tp + fn) if (tp + fn) != 0 else 1
        f1_score = 2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp +
                                                       fn) != 0 else 1

        return precision, recall, f1_score, tp_box, fp_box, fn_box

    def get_KD_acc_batch(self, batch_ind, res):
        """calculate precision, recall and f1 score for image batch ind,
        the metrics are over number of images per batch

        Args:
            batch_ind (int): batch image index (NOT per image index)
            res (list of list): detection result for batch images
        """

        ind_start = batch_ind * self.batch_size
        # always return the same number of results as in res,
        # some videos has no objectes detected in ending frames,
        # so the entry in ground truth is missing
        ind_end = ind_start + len(res)
        ind_list = list(range(ind_start, ind_end))

        precision_all = []
        recall_all = []
        f1_score_all = []
        tp_box_all = []
        fp_box_all = []
        fn_box_all = []
        for i, ind in enumerate(ind_list):
            precision, recall, f1_score, tp_box, fp_box, fn_box = \
                self.get_KD_acc(ind, res[i])
            precision_all.append(precision)
            recall_all.append(recall)
            f1_score_all.append(f1_score)
            tp_box_all.append(tp_box)
            fp_box_all.append(fp_box)
            fn_box_all.append(fn_box)

        return np.mean(precision_all), np.mean(recall_all), np.mean(
            f1_score_all), tp_box_all, fp_box_all, fn_box_all
