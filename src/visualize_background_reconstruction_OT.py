import time
from pathlib import Path

import cv2
import numpy as np

from utils.video_source import FromRawVideo
from utils.misc import concat_image
from models.yolov3_trt.yolov3 import Yolov3

vid_path = '../dataset/youtube/crossroad2.mp4'
vid_name = Path(vid_path).stem
dataset = FromRawVideo(vid_path, resize=None)
sub = cv2.createBackgroundSubtractorKNN()
t_last = time.time()
model = Yolov3()
height, width = None, None
bg = None

cv2.namedWindow('background reconstruction', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('background reconstruction', cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)

for ind, frame in enumerate(dataset.read()):

    # perform object detection
    if height is None:
        height, width = frame.shape[:2]
    mask_old = np.zeros((height, width), dtype='uint8')
    res = model.infer(frame)
    for label, conf, (x1, y1, x2, y2) in res:
        mask_old[y1:y2, x1:x2] = 1

    if bg is not None:
        # for false positive roi, use new frame
        mask_new = 1 - mask_old
        part_bg = cv2.bitwise_and(bg, bg, mask=mask_old)
        part_fg = cv2.bitwise_and(frame, frame, mask=mask_new)
        part_all = cv2.add(part_bg, part_fg)
    else:
        part_all = frame

    t1 = time.time()
    mask = sub.apply(part_all, learningRate=-1)
    bg = sub.getBackgroundImage()
    t2 = time.time()
    fps = 1 / (t2 - t1)

    img = concat_image({
        'none': frame,
        'edited': part_all,
        f'#{ind}, {fps:.1f} fps': bg,
        'mask': mask
    })

    while True:
        t_now = time.time()
        if t_now - t_last > 0.03:
            cv2.imshow('background reconstruction', img)
            t_last = t_now
            break
        else:
            time.sleep(0.001)

    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
    elif key == 32:  # hit space to pause
        cv2.waitKey(0)
