import numpy as np
import cv2

from utils.misc import dilate


class RoiParams:
    def __init__(self, height, width, num_roi=8, keep_ratio=0.94):
        self.width = width
        self.height = height
        self.num_roi = num_roi  # maximum number of ROI regions to set
        # leave some pixels out when setting roi regions, required on jetson
        self.keep_ratio = keep_ratio
        self.qp_large = 15
        self.qp_small = 0

        # size_bb to record history bounding box size for each pixel
        self.size_bb = np.zeros((self.height, self.width))

    def update_size_bb(self, bbs, update_rate=0.2):
        """update size_bb matrix

        Args:
            bbs: list of bounding box detection
        """
        if len(bbs) == 0:
            return

        for (_, x1, y1, x2, y2) in bbs:
            size = (x2 - x1) * (y2 - y1)
            self.size_bb[y1:y2, x1:x2] = (1 - update_rate) * self.size_bb[
                y1:y2, x1:x2] + update_rate * size

    def send_roi_params(self, communicator, th_size=5000):
        mask_small = np.zeros((self.height, self.width))
        mask_small[(self.size_bb > 10) & (self.size_bb <= th_size)] = 1
        mask_small = cv2.morphologyEx(
            mask_small, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (16, 16)))
        contours, _ = cv2.findContours(mask_small.astype('uint8'),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        region_small = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # remove regions that are too small
            if w * h >= 50 * 50:
                region_small.append([x, y, x + w, y + h])

        # merge overlapped bb
        i = 0
        while i < len(region_small) - 1:
            j = i + 1
            overlapped = False
            while j < len(region_small):
                x1_max = max(region_small[i][0], region_small[j][0])
                x2_min = min(region_small[i][2], region_small[j][2])
                y1_max = max(region_small[i][1], region_small[j][1])
                y2_min = min(region_small[i][3], region_small[j][3])
                if x1_max > x2_min or y1_max > y2_min:
                    # no overlap
                    j += 1
                else:
                    # overlapped
                    overlapped = True
                    x1_min = min(region_small[i][0], region_small[j][0])
                    x2_max = max(region_small[i][2], region_small[j][2])
                    y1_min = min(region_small[i][1], region_small[j][1])
                    y2_max = max(region_small[i][3], region_small[j][3])
                    region_small[j] = [x1_min, y1_min, x2_max, y2_max]
                    del region_small[i]
                    break

            if not overlapped:
                i += 1

        mask_large = np.zeros((self.height, self.width))
        mask_large[self.size_bb > th_size] = 1
        contours, _ = cv2.findContours(mask_large.astype('uint8'),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        region_large = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # remove regions that are too small
            if w * h >= 50 * 50:
                region_large.append([x, y, x + w, y + h])

        # merge overlapped bb
        i = 0
        while i < len(region_large) - 1:
            j = i + 1
            overlapped = False
            while j < len(region_large):
                x1_max = max(region_large[i][0], region_large[j][0])
                x2_min = min(region_large[i][2], region_large[j][2])
                y1_max = max(region_large[i][1], region_large[j][1])
                y2_min = min(region_large[i][3], region_large[j][3])
                if x1_max > x2_min or y1_max > y2_min:
                    # no overlap
                    j += 1
                else:
                    # overlapped
                    overlapped = True
                    x1_min = min(region_large[i][0], region_large[j][0])
                    x2_max = max(region_large[i][2], region_large[j][2])
                    y1_min = min(region_large[i][1], region_large[j][1])
                    y2_max = max(region_large[i][3], region_large[j][3])
                    region_large[j] = [x1_min, y1_min, x2_max, y2_max]
                    del region_large[i]
                    break

            if not overlapped:
                i += 1

        # check num_roi requirements
        if len(region_small) + len(region_large) > self.num_roi:
            # merge bbs in region_large into one
            x1_min = min([each[0] for each in region_large])
            x2_max = max([each[2] for each in region_large])
            y1_min = min([each[1] for each in region_large])
            y2_max = max([each[3] for each in region_large])
            region_large = [[x1_min, y1_min, x2_max, y2_max]]
            while len(region_small) > self.num_roi - 1:
                region_small = sorted(region_small, key=lambda x: x[0])
                x1_min = min(region_small[0][0], region_small[1][0])
                x2_max = max(region_small[0][2], region_small[1][2])
                y1_min = min(region_small[0][1], region_small[1][1])
                y2_max = max(region_small[0][3], region_small[1][3])
                region_small[1] = [x1_min, y1_min, x2_max, y2_max]
                del region_small[0]

        # check keep_ratio requirements
        area_small = sum([(each[2] - each[0]) * (each[3] - each[1])
                          for each in region_small])
        area_large = sum([(each[2] - each[0]) * (each[3] - each[1])
                          for each in region_large])
        region_large = sorted(region_large,
                              key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        diff = area_small + area_large - self.width * self.height * self.keep_ratio
        while diff > 0 and len(region_large) > 0:
            if (region_large[0][2] - region_large[0][0]) * (
                    region_large[0][3] - region_large[0][1]) > diff:
                region_large[0][2] = int(
                    region_large[0][0] + diff //
                    (region_large[0][3] - region_large[0][1]))
                break
            else:
                del region_large[0]
                region_large = sorted(region_large,
                                      key=lambda x: (x[2] - x[0]) *
                                      (x[3] - x[1]))
                diff = area_small + area_large - self.width * self.height * self.keep_ratio

        line = str(len(region_small) + len(region_large))
        for (x1, y1, x2, y2) in region_large:
            line += f' {self.qp_large} {x1} {y1} {x2-x1} {y2-y1}'
        for (x1, y1, x2, y2) in region_small:
            line += f' {self.qp_small} {x1} {y1} {x2-x1} {y2-y1}'
        size = communicator.send_roi_params(line)
        return region_small, region_large, size

    def send_det_roi_params(self, communicator, bbox, th_size=300):
        mask = np.zeros((self.height, self.width))
        for _, x1, y1, x2, y2 in bbox:
            mask[y1:y2, x1:x2] = 1
        mask = dilate(mask, 15)
        mask[(self.size_bb > 10) & (self.size_bb <= th_size)] = 1
        line = self.mask2roiparam(mask)
        size = communicator.send_roi_params(line)
        return [], [], size

    def mask2roiparam(self, mask_small):
        num_roi = 8  # maximum number of ROI regions to set
        # leave some pixels out when setting roi regions, required on jetson
        keep_ratio = 0.94
        qp_large = 30
        qp_small = 0

        mask_small = cv2.morphologyEx(
            mask_small, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (16, 16)))
        contours, _ = cv2.findContours(mask_small.astype('uint8'),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        region_small = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # remove regions that are too small
            if w * h >= 50 * 50:
                region_small.append([x, y, x + w, y + h])

        # merge overlapped bb
        i = 0
        while i < len(region_small) - 1:
            j = i + 1
            overlapped = False
            while j < len(region_small):
                x1_max = max(region_small[i][0], region_small[j][0])
                x2_min = min(region_small[i][2], region_small[j][2])
                y1_max = max(region_small[i][1], region_small[j][1])
                y2_min = min(region_small[i][3], region_small[j][3])
                if x1_max > x2_min or y1_max > y2_min:
                    # no overlap
                    j += 1
                else:
                    # overlapped
                    overlapped = True
                    x1_min = min(region_small[i][0], region_small[j][0])
                    x2_max = max(region_small[i][2], region_small[j][2])
                    y1_min = min(region_small[i][1], region_small[j][1])
                    y2_max = max(region_small[i][3], region_small[j][3])
                    region_small[j] = [x1_min, y1_min, x2_max, y2_max]
                    del region_small[i]
                    break

            if not overlapped:
                i += 1

        mask_large = 1 - mask_small
        contours, _ = cv2.findContours(mask_large.astype('uint8'),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        region_large = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # remove regions that are too small
            if w * h >= 50 * 50:
                region_large.append([x, y, x + w, y + h])

        # merge overlapped bb
        i = 0
        while i < len(region_large) - 1:
            j = i + 1
            overlapped = False
            while j < len(region_large):
                x1_max = max(region_large[i][0], region_large[j][0])
                x2_min = min(region_large[i][2], region_large[j][2])
                y1_max = max(region_large[i][1], region_large[j][1])
                y2_min = min(region_large[i][3], region_large[j][3])
                if x1_max > x2_min or y1_max > y2_min:
                    # no overlap
                    j += 1
                else:
                    # overlapped
                    overlapped = True
                    x1_min = min(region_large[i][0], region_large[j][0])
                    x2_max = max(region_large[i][2], region_large[j][2])
                    y1_min = min(region_large[i][1], region_large[j][1])
                    y2_max = max(region_large[i][3], region_large[j][3])
                    region_large[j] = [x1_min, y1_min, x2_max, y2_max]
                    del region_large[i]
                    break

            if not overlapped:
                i += 1

        # check num_roi requirements
        if len(region_small) + len(region_large) > num_roi:
            # merge bbs in region_large into one
            x1_min = min([each[0] for each in region_large])
            x2_max = max([each[2] for each in region_large])
            y1_min = min([each[1] for each in region_large])
            y2_max = max([each[3] for each in region_large])
            region_large = [[x1_min, y1_min, x2_max, y2_max]]
            while len(region_small) > num_roi - 1:
                region_small = sorted(region_small, key=lambda x: x[0])
                x1_min = min(region_small[0][0], region_small[1][0])
                x2_max = max(region_small[0][2], region_small[1][2])
                y1_min = min(region_small[0][1], region_small[1][1])
                y2_max = max(region_small[0][3], region_small[1][3])
                region_small[1] = [x1_min, y1_min, x2_max, y2_max]
                del region_small[0]

        # check keep_ratio requirements
        area_small = sum([(each[2] - each[0]) * (each[3] - each[1])
                          for each in region_small])
        area_large = sum([(each[2] - each[0]) * (each[3] - each[1])
                          for each in region_large])
        region_large = sorted(region_large,
                              key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        diff = area_small + area_large - self.width * self.height * keep_ratio
        while diff > 0 and len(region_large) > 0:
            if (region_large[0][2] - region_large[0][0]) * (
                    region_large[0][3] - region_large[0][1]) > diff:
                region_large[0][2] = int(
                    region_large[0][0] + diff //
                    (region_large[0][3] - region_large[0][1]))
                break
            else:
                del region_large[0]
                region_large = sorted(region_large,
                                      key=lambda x: (x[2] - x[0]) *
                                      (x[3] - x[1]))
                diff = area_small + area_large - self.width * self.height * keep_ratio

        line = str(len(region_small) + len(region_large))
        for (x1, y1, x2, y2) in region_large:
            line += f' {qp_large} {x1} {y1} {x2-x1} {y2-y1}'
        for (x1, y1, x2, y2) in region_small:
            line += f' {qp_small} {x1} {y1} {x2-x1} {y2-y1}'
        return line
