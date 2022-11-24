import os
import queue
import threading
import time
from pathlib import Path

import cv2

from models.openpifpaf_trt.openpifpaf import Openpifpaf
from utils.video_source import FromImgDir

video_path = '../dataset/softbio_vid.mp4'
video_name = Path(video_path).stem
dataset = FromImgDir(
    dst_dir='../dataset/human36m/s_01_act_11_subact_01_ca_02/')
print(f'Total: {len(dataset)} frames')
model = Openpifpaf()
batch_size = 15
q = queue.Queue()
time_all = []


def infer():
    ind = -1

    for _, imgs in dataset.read_batch(batch_size):

        t1 = time.time()
        preds = model.infer_batch(imgs)
        t2 = time.time()
        time_elapsed = t2 - t1
        print(f'{time_elapsed*1000:.2f} ms for {len(imgs)} images')
        time_all.append(time_elapsed)

        for img, pred in zip(imgs, preds):
            ind += 1
            q.put({
                'ind': ind,
                'img': img,
                'pred': pred,
                'time_elapsed': time_elapsed
            })


def show():
    t_last = time.time()
    for dic in iter(q.get, None):
        ind = dic['ind']
        img = dic['img']
        pred = dic['pred']
        time_elapsed = dic['time_elapsed']

        color = (0, 255, 255)
        img = cv2.resize(img, (641, 369))
        for i, pred_object in enumerate(pred):
            pred = pred_object.data
            pred_visible = pred[pred[:, 2] > 0]
            xs = pred_visible[:, 0]
            ys = pred_visible[:, 1]
            for x, y in zip(xs, ys):
                cv2.circle(img, ((int)(x), (int)(y)), 2, color, -1)
            decode_order = [(a, b)
                            for (a, b, c, d) in pred_object.decoding_order]
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
                cv2.imshow(video_name, img)
                t_last = t_cur
                break
            else:
                time.sleep(0.001)

        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            os._exit(1)
        elif key == 32:  # hit space to pause
            cv2.waitKey(0)


if __name__ == '__main__':
    t1 = threading.Thread(target=infer, args=())
    t2 = threading.Thread(target=show, args=())
    t1.start()
    t2.start()
    t1.join()
    t2.join()
