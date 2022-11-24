import cv2
import numpy as np

from utils.misc import get_logger

logger = get_logger(name='cloud', level='DEBUG')


class BackgroundReconstructor:
    def __init__(self, thresh_roi_black, diff_th=15):
        """
        Args:
            thresh_roi_black (int): threshold to decide whether the region is bg
            diff_th (int): threshold to decide whether the bg is changing
        """
        self.thresh_roi_black = thresh_roi_black
        self.diff_th = diff_th
        self.sub = cv2.createBackgroundSubtractorKNN()
        self.bg = None
        self.height, self.width = None, None
        self.last_sent_bg = None
        self.last_sent_ind = 0
        self.save_ind = 1

    def update(self, ind, frame, res):
        if self.height is None:
            self.height, self.width = frame.shape[:2]

        # number of pixels with objects
        n_detected = 0

        # mask for objects
        mask_bg = np.zeros((self.height, self.width), dtype='uint8')
        for _, x1, y1, x2, y2 in res:
            n_detected += (x2 - x1) * (y2 - y1)
            mask_bg[y1:y2, x1:x2] = 1

        # mask of blacked region
        mask_zero = (frame.astype('float').sum(axis=2) <
                     self.thresh_roi_black).astype('uint8')

        # for blacked region and detected bboxes, use existing bg
        mask_bg = cv2.bitwise_or(mask_bg, mask_zero)

        # number of pixels in roi
        n_roi = max(1, cv2.countNonZero(frame.sum(axis=2)))

        # ratio of correct roi pixels
        detect_ratio = n_detected / n_roi

        # if too many false positive roi pixels
        if detect_ratio < 0.9:
            if self.bg is not None:
                # for false positive roi, use new frame
                # combining bg and fg is necessary since the roi only
                # contains part of the image
                mask_fg = 1 - mask_bg
                part_bg = cv2.bitwise_and(self.bg, self.bg, mask=mask_bg)
                part_fg = cv2.bitwise_and(frame, frame, mask=mask_fg)
                part_joined = cv2.add(part_bg, part_fg)
            else:
                part_joined = frame
            self.save_ind += 1
            _ = self.sub.apply(part_joined)
            self.bg = self.sub.getBackgroundImage()

    def check_and_send_bg(self, ind, communicator):
        # skip the first two learned background images
        size = 0
        if ind < 2 or self.bg is None:
            return size

        if self.last_sent_bg is None:
            size = communicator.send_bg(self.bg)
            self.last_sent_bg = self.bg
            self.last_sent_ind = ind
            logger.debug(f'sending bg on ind {ind} ({size/1024:.2f} KB)')

        else:
            diff = cv2.absdiff(self.last_sent_bg, self.bg).mean()
            if diff > self.diff_th and ind > self.last_sent_ind + 10:
                size = communicator.send_bg(self.bg)
                self.last_sent_bg = self.bg
                self.last_sent_ind = ind
                logger.debug(f'sending bg on ind {ind} ({size/1024:.2f} KB)')

        return size

    def get_bg(self):
        if self.bg is None:
            return None
        else:
            return self.bg.copy()
