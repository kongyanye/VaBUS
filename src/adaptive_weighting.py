import numpy as np


class AdaptiveWeighting:
    """Adaptive weighting module learns a weight in the range [0, 10].
    The idea is to set lower weight for regions that are less likely to
    exist an object. When there's no object found and pixels are seldomly
    changing, the weight will decay slowly.
    """
    def __init__(self, macro_size, th_std=3, win_size=15):
        # macro size of super pixel
        self.macro_size = macro_size
        # threshold to decide whether a pixel is substantially changed
        self.th_std = th_std
        # number of frames used to calculate std for each pixel
        self.win_size = win_size

        # count matrix for each macro pixel, used to calculate weight
        self.count_mat = None
        # height and width after conv2d
        self.height_c, self.width_c = None, None
        # matrx to hold the last win_size pixels for each pixel
        self.his = None
        # index to update the next value in self.his
        self.modify_ind = None
        # function to transform count into weight
        self.weight_fn = lambda x: np.clip(1 - x / 100, 0, 4)

    def update(self, fg_bboxes_batch, overlay_bboxes=[]):
        """update count matrix with mask
        count_mat decrease where objects exist, otherwise increase

        Args:
            fg_bboxes_batch: batches of bounding box detections
            overlay_bboxes: list of bounding box for overlay objects
        """

        if self.count_mat is None:
            return

        # for each image
        for fg_bboxes in fg_bboxes_batch:

            # count increase for pixels without objects
            self.count_mat += 1

            # count decrease for pixels with objects
            for _, x1, y1, x2, y2 in fg_bboxes:
                nx1, ny1, nx2, ny2 = list(
                    map(self.transform_c, [x1, y1, x2, y2]))
                self.count_mat[ny1:ny2, nx1:nx2] -= 16

        self.count_mat = np.clip(self.count_mat, -300, 300)

        # reset count for overlay pixels
        for _, x1, y1, x2, y2 in overlay_bboxes:
            nx1, ny1, nx2, ny2 = list(map(self.transform_c, [x1, y1, x2, y2]))
            self.count_mat[ny1:ny2, nx1:nx2] = 0

    def get_weight(self, img_smooth):
        """get weight matrix for current batch of images

        Args:
            img_smooth: batch images of shape (batch_size, H, W)
        """
        # take mean over batch
        img_smooth = img_smooth.mean(axis=0)

        # intialize height_c and width_c
        if self.height_c is None:
            self.height_c, self.width_c = img_smooth.shape

        # initialize count_mat, his and modify_ind
        if self.count_mat is None:
            self.count_mat = np.zeros((self.height_c, self.width_c))
            self.his = np.empty((self.height_c, self.width_c, self.win_size))
            self.modify_ind = 0

        # calculate std for pixels
        self.modify_ind = self.modify_ind % self.win_size
        self.his[:, :, self.modify_ind] = img_smooth
        self.modify_ind += 1

        # calculate std for all non-NaN values
        with np.errstate(over='ignore'):
            std = np.nanstd(self.his, axis=2)

        # reset pixels that have substantially changed
        self.count_mat[(std >= self.th_std) & (self.count_mat > 0)] = 0

        # calculate weight based on count
        weight = self.weight_fn(self.count_mat)
        return weight

    def transform_c(self, x):
        ret = (x - self.macro_size + 1) / self.macro_size
        if int(ret) == ret:
            return int(ret)
        else:
            return int(ret) + 1
