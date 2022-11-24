import time

import cv2
import pandas as pd

from utils.video_source import FromImgDir

dataset = FromImgDir(dst_dir='../dataset/mupots3d/TS11/', resize=(406, 720))
print(f'Total: {len(dataset)} frames')
gt = pd.read_csv('../dataset/ground_truth/KD_resize_406_720/TS11.csv')
gt['x'] = gt['x'] / 641 * 720
gt['y'] = gt['y'] / 369 * 406

color = (0, 255, 0)
skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
            (1, 3), (2, 4), (3, 5), (4, 6)]
cv2.namedWindow('GT', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('GT', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

t_last = time.time()
for ind, img in dataset.read():
    t_cur = time.time()

    rows, cols, _ = img.shape
    res = gt[gt['ind'] == ind]
    human_inds = res['human_ind'].unique()

    for human_ind in human_inds:
        res_human = res[res['human_ind'] == human_ind]
        kp_inds = set(res_human['kp_ind'].values)
        for i in range(len(res_human)):
            x, y = res_human.iloc[i, [3, 4]].values.astype('int')
            kp_ind = res_human.iloc[i, 2]
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            _ = cv2.putText(img, str(kp_ind + 1), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (112, 217, 119), 1)
        for kp_ind1, kp_ind2 in skeleton:
            if kp_ind1 in kp_inds and kp_ind2 in kp_inds:
                x1, y1 = res_human.loc[res['kp_ind'] == kp_ind1,
                                       ['x', 'y']].values.astype('int')[0]
                x2, y2 = res_human.loc[res['kp_ind'] == kp_ind2,
                                       ['x', 'y']].values.astype('int')[0]
                cv2.line(img, (x1, y1), (x2, y2), color, 1)

    t_cur = time.time()
    t_left = 0.033 - (t_cur - t_last)
    if t_left > 0:
        time.sleep(t_left)
    t_last = t_cur
    cv2.imshow('GT', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
    elif key == 32:  # hit space to pause
        cv2.waitKey(0)

cv2.destroyAllWindows()
