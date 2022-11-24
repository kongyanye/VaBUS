import os
import time
from pathlib import Path

import cv2
import numpy as np

from utils.video_source import FromRawVideo

vid_path = '../dataset/youtube/city.mp4'
vid_name = Path(vid_path).stem
dataset = FromRawVideo(vid_path, resize=None)
sub = cv2.createBackgroundSubtractorKNN()
t_last = time.time()
t_used = []

for ind, frame in enumerate(dataset.read()):
    t1 = time.time()
    mask = sub.apply(frame)
    back = sub.getBackgroundImage()
    t2 = time.time()
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=kernel)

    t_used.append(int((t2 - t1) * 1000))
    fps = 1 / (t2 - t1)
    print(f'time used: {(t2-t1)*1000:.1f} ms')

    img1 = np.concatenate((frame, back), axis=0)
    img2 = np.concatenate((np.dstack(
        (mask, ) * 3, ), np.zeros(
            (mask.shape[0], mask.shape[1], 3)).astype('uint8')),
                          axis=0)
    img = np.concatenate((img1, img2), axis=1)
    img = cv2.resize(img, (1920, 1080))
    _ = cv2.putText(img, f'#{ind}, {fps:.1f} fps', (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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

        # plot time used
        # can't import cv2 and matplotlib the same time
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 12))
        _ = plt.hist(t_used, bins=100, range=(0, 100))
        plt.xlabel('time per frame')
        plt.ylabel('count')
        plt.title(f'KNN Background Subtractor ({vid_name}, num={len(t_used)})')
        if not os.path.exists('../results'):
            os.mkdir('../results')
        plt.savefig(f'../results/KNN_background_subtractor_{vid_name}.jpg')

        break
    elif key == 32:  # hit space to pause
        cv2.waitKey(0)
