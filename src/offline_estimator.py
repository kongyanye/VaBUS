import numpy as np


class OfflineEstimator:
    def __init__(self,
                 height,
                 width,
                 batch_size,
                 update_rate=0.7,
                 enlarge_ratio=1.02):

        self.height, self.width = height, width
        self.batch_size = batch_size
        self.update_rate = update_rate
        self.enlarge_ratio = enlarge_ratio

        self.shift_x = np.zeros((height, width))
        self.shift_y = np.zeros((height, width))
        self.expand = np.ones((height, width))
        self.last_track = None

    def update(self, batch_track):
        # update shift and expand
        if self.last_track is not None:
            for cur_track in batch_track:
                for ind in cur_track:
                    if ind not in self.last_track:
                        continue
                    else:
                        cx1, cy1, cx2, cy2, _ = cur_track[ind]
                        cx1, cy1, cx2, cy2 = list(
                            map(int, [cx1, cy1, cx2, cy2]))
                        px1, py1, px2, py2, _ = self.last_track[ind]
                        px1, py1, px2, py2 = list(
                            map(int, [px1, py1, px2, py2]))
                        prev_mid_x = (px1 + px2) / 2
                        prev_mid_y = (py1 + py2) / 2
                        cur_mid_x = (cx1 + cx2) / 2
                        cur_mid_y = (cy1 + cy2) / 2
                        self.shift_x[
                            py1:py2,
                            px1:px2] = (1 - self.update_rate) * self.shift_x[
                                py1:py2, px1:px2] + self.update_rate * (
                                    cur_mid_x - prev_mid_x)
                        self.shift_y[
                            py1:py2,
                            px1:px2] = (1 - self.update_rate) * self.shift_y[
                                py1:py2, px1:px2] + self.update_rate * (
                                    cur_mid_y - prev_mid_y)
                        ratio = (cx2 - cx1) / (px2 - px1) * (cy2 - cy1) / (
                            py2 - py1) * self.enlarge_ratio
                        self.expand[
                            py1:py2,
                            px1:px2] = (1 - self.update_rate) * self.expand[
                                py1:py2, px1:px2] + self.update_rate * ratio
                self.last_track = cur_track

        if len(batch_track) > 0:
            self.last_track = batch_track[-1]

    def suppress_xy(self, i, x):
        return x * 0.90**i

    def suppress_ex(self, i, ex):
        if ex > 1:
            return max(ex * 0.99**i, 1)
        else:
            return min(ex * 1.01**i, 1)

    def estimate(self, last_res, num, mv_det):
        """Generate a batch of estimated results based on last infer result

        Args:
            last_res: list of [label, x1, y1, x2, y2], results of last frame
            num: number of frames to estimate
        """
        shift_x = self.shift_x.copy()
        shift_y = self.shift_y.copy()
        expand = self.expand.copy()

        nres_batch = []  # results of a batch of frames
        for i in range(num):
            nres = []  # results of a frame
            for res in last_res:
                label, x1, y1, x2, y2 = res
                x1, y1, x2, y2 = list(map(int, [x1, y1, x2, y2]))
                sx = np.mean(shift_x[y1:y2, x1:x2])
                sy = np.mean(shift_y[y1:y2, x1:x2])
                ex = np.mean(expand[y1:y2, x1:x2])

                # suppress sx, sy and ex
                sx = 0.7 * self.suppress_xy(i, sx)
                sy = 0.7 * self.suppress_xy(i, sy)
                ex = self.suppress_ex(i, ex)

                # expand on shifted direction
                nx1 = int(x1 + sx - (ex - 1) * (x2 - x1) / 2)
                nx2 = int(x2 + sx + (ex - 1) * (x2 - x1) / 2)
                ny1 = int(y1 + sy - (ex - 1) * (y2 - y1) / 2)
                ny2 = int(y2 + sy + (ex - 1) * (y2 - y1) / 2)

                # clip x and y
                nx1, nx2 = np.clip([nx1, nx2], 0, self.width - 1)
                ny1, ny2 = np.clip([ny1, ny2], 0, self.height - 1)
                if (nx1 - nx2) * (ny1 - ny2) != 0:
                    nres.append([label, nx1, ny1, nx2, ny2])
            if i < num-1:
                last_res = mv_det[i+1]
            nres_batch.append(nres)
        return nres_batch

