import os
import time

import cv2

from models.openpifpaf_trt.openpifpaf import Openpifpaf
from utils.video_source import FromImgDir

dataset = FromImgDir(dst_dir='../dataset/mupots3d/TS11/', resize=(406, 720))
print(f'Total: {len(dataset)} frames')
model = Openpifpaf()

t_last = time.time()
time_all = []
for ind, img in dataset.read():
    rows, cols, _ = img.shape
    t1 = time.time()
    pred = model.infer(img)
    t2 = time.time()
    time_elapsed = t2 - t1
    time_all.append(time_elapsed)

    color = (0, 255, 255)
    img = cv2.resize(img, (641, 369))
    for i, pred_object in enumerate(pred):
        pred = pred_object.data
        pred_visible = pred[pred[:, 2] > 0]
        xs = pred_visible[:, 0]
        ys = pred_visible[:, 1]
        for x, y in zip(xs, ys):
            cv2.circle(img, ((int)(x), (int)(y)), 2, color, -1)
        decode_order = [(a, b) for (a, b, c, d) in pred_object.decoding_order]
        for index, (a, b) in enumerate(decode_order):
            if (a + 1, b + 1) in pred_object.skeleton or (
                    b + 1, a + 1) in pred_object.skeleton:
                x1, y1, _ = pred_object.decoding_order[index][2]
                x2, y2, _ = pred_object.decoding_order[index][3]
            else:
                continue
            cv2.line(img, ((int)(x1), (int)(y1)), ((int)(x2), (int)(y2)),
                     color, 2)

    _ = cv2.putText(img, f'#{ind}, {time_elapsed*1000:.1f} ms', (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (112, 217, 119), 2)

    while True:
        t_cur = time.time()
        if t_cur - t_last >= 0.033:
            cv2.imshow(dataset.name, img)
            t_last = t_cur
            break
        else:
            time.sleep(0.001)

    key = cv2.waitKey(1)
    if key == ord('q'):
        print(f'average speed: {len(time_all)/sum(time_all):.1f} fps')
        cv2.destroyAllWindows()
        os._exit(1)
    elif key == 32:  # hit space to pause
        cv2.waitKey(0)

cv2.destroyAllWindows()
os._exit(1)
