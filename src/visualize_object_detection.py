import os
import time
from pathlib import Path

import cv2

# from models.fasterrcnn.fasterrcnn_resnet101 import FasterRCNN_ResNet101
from models.yolov3_trt.yolov3 import Yolov3
# from models.efficientdet import efficientdet
from utils.video_source import FromRawVideo

video_path = '../dataset/youtube/city.mp4'
video_name = Path(video_path).stem
dataset = FromRawVideo(src_file=video_path, resize=(406, 720))
model = Yolov3()

t_last = time.time()
time_all = []
for ind, img in dataset.read():

    rows, cols, _ = img.shape
    t1 = time.time()
    res = model.infer(img)
    t2 = time.time()
    time_elapsed = t2 - t1
    time_all.append(time_elapsed)

    for label, conf, (x1, y1, x2, y2) in res:
        if conf < 0.5:
            continue
        print(label, conf, (x1, y1, x2, y2))
        if label == 'persons':
            color = (117, 0, 255)  # bgr red
        else:
            color = (113, 248, 249)  # bgr yellow
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        _ = cv2.putText(img, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

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
        print(f'average speed: {len(time_all)/sum(time_all):.1f} fps')
        os._exit(1)
    elif key == 32:  # hit space to pause
        cv2.waitKey(0)
