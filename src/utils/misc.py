import logging
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import time


def get_logger(name, level='INFO'):
    logging.basicConfig(
        format="[%(name)s #%(lineno)s] %(asctime)s %(levelname)s"
        " -- %(message)s",
        level=level)

    logger = logging.getLogger(name)
    logger.addHandler(logging.NullHandler())

    return logger


def conv2d(image, kernel_value, kernel_size, stride, channel_num=1, padding=0):
    """convolution of single or batches of grayscale image

    Args:
        image (np.ndarray): single or batches of image, the shape must be
            one of [H, W, 3], [H, W], [batch_size, H, W, 3] or
            [batch_size, H, W]
        kernel (np.ndarray): conv kernel
        stride (int): conv stride
        padding (int): conv padding
        channel_num (int): number of image channels, 1 for grayscale
            image, 3 for RGB image

    Returns:
        convolution result of shape [batch_size, H, W] or [batch_size, 3, H, W]
    """
    assert channel_num in [1, 3], ('channel_num must be in [1, 3],'
                                   f'got {channel_num}')

    if image.dtype != 'float':
        image = image.astype('float')

    # grayscale image
    if channel_num == 1:
        if len(image.shape) == 2:
            # single image conv
            image = torch.tensor(np.expand_dims(image, axis=(0, 1)))
        else:
            # batch image conv
            image = torch.tensor(np.expand_dims(image, axis=1))
        kernel = torch.tensor(
            np.ones((1, 1, kernel_size, kernel_size)) * kernel_value)
    # RGB image
    else:
        if len(image.shape) == 3:
            # single image conv
            image = torch.tensor(np.expand_dims(image, axis=0))
        else:
            image = torch.tensor(image)
        # switch channel to the second axis
        image = np.transpose(image, (0, 3, 1, 2))
        kernel = torch.tensor(
            np.ones((1, 3, kernel_size, kernel_size)) * kernel_value)
    out = F.conv2d(image, kernel, stride=stride, padding=padding)
    return out.numpy().squeeze()


def dilate(mask, kernel_size):
    """dilate single or batches of binary mask

    Args:
        mask: single or batches of binary mask of [H, W]
        kernel_size: size of dilation kernel, must be odd number
    """
    if len(mask.shape) == 2:
        # single image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel)
    else:
        # batch image
        assert kernel_size % 2 == 1, ('kernel size must be odd number,'
                                      f'got {kernel_size}')

        mask = conv2d(image=mask,
                      kernel_value=1,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=(kernel_size - 1) // 2,
                      channel_num=1)
        mask[mask > 0] = 1
    return mask


def resample(image, size):
    """resample single or batches of grayscale image, upsample
    if size if larger than original image else downsample

    Args:
        image: single or batches of grayscale image of [H, W]
        size: target size of [H_out, W_out]
    """

    if len(image.shape) == 2:
        # single image
        if size == image.shape[:2]:
            return image
        if image.dtype != 'float':
            image = image.astype('float')

        image = torch.tensor(np.expand_dims(image, axis=(0, 1)))
        out = F.interpolate(image, size=size)
        return out.numpy().squeeze()
    else:
        # batch image
        if size == image[0].shape[:2]:
            return image
        if image.dtype != 'float':
            image = image.astype('float')

        image = torch.tensor(np.expand_dims(image, axis=1))
        out = F.interpolate(image, size=size)
        return out.numpy().squeeze()


def concat_image(dic, size=None):
    """concat image and add captions for cv2.imshow

    Args:
        dic (Dict of caption (key) and image (value)): e.g.: {'background':
            bg}, set caption to 'none' to disable caption display
        size (tuple of (int, int)): target image size of [height, width],
            if None, assert all images are of the same size
    """
    if size is not None:
        height, width = size
    imgs = []
    for cap, img in dic.items():
        if size is None:
            assert img is not None, f'the first img can not be None when \
            size is None, but got {cap} to be None'

            height, width = img.shape[:2]
        if img is None:
            img = np.zeros((height, width, 3), dtype='uint8')
        else:
            if img.dtype != 'uint8':
                img = img.astype('uint8')
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            if len(img.shape) == 2:
                img = np.dstack((img, ) * 3)
        if cap != 'none':
            _ = cv2.putText(img, cap, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)
        imgs.append(img)

    # calculate the number of rows and cols to display
    cols = np.sqrt(len(imgs))
    if cols % 1 != 0:
        cols = int(cols) + 1
    else:
        cols = int(cols)
    rows = len(imgs) / cols
    if rows % 1 != 0:
        rows = int(rows) + 1
    else:
        rows = int(rows)

    # concat all into one
    imgs_row = []
    for i in range(rows):
        start = i * cols
        end = (i + 1) * cols
        if end > len(imgs):
            tmp = np.hstack(
                np.concatenate((imgs[start:len(imgs)],
                                np.zeros(((end - len(imgs)), height, width, 3),
                                         dtype='uint8'))))
        else:
            tmp = np.hstack(imgs[start:end])
        imgs_row.append(tmp)
    imgs_all = np.vstack(imgs_row)
    return imgs_all


def timing(func, *args, **kwargs):
    def func_proxy(*args, **kwargs):
        t1 = time.time()
        ret = func(*args, **kwargs)
        t2 = time.time()
        return t2 - t1, ret

    return func_proxy


def wrapc(text, c='r'):
    colors = {
        'r': '\033[1;031m',  # red
        'g': '\033[1;032m',  # green
        'o': '\033[1;033m',  # orange
        'p': '\033[1;035m',  # purple
        'w': '\033[1;037m',  # white
        'y': '\033[1;093m'  # yellow
    }

    start = colors.get(c, 'w')
    end = '\033[0m'
    return f'{start}{text}{end}'


def nms_filter(boxes, nms_threshold=0.5):
    """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding
    boxes and return the filtered boxes.

    Args:
        boxes: a list containing N bounding-box coordinates with 5 columns
            (label,x1,y1,x2,y2) or 4 columns (x1,y1,x2,y2)
    """
    if len(boxes) == 0:
        return []

    boxes_np = np.array(boxes)

    if boxes_np.shape[1] == 5:
        x1 = boxes_np[:, 1].astype('int')
        y1 = boxes_np[:, 2].astype('int')
        x2 = boxes_np[:, 3].astype('int')
        y2 = boxes_np[:, 4].astype('int')
    else:
        x1 = boxes_np[:, 0].astype('int')
        y1 = boxes_np[:, 1].astype('int')
        x2 = boxes_np[:, 2].astype('int')
        y2 = boxes_np[:, 3].astype('int')

    areas = (x2 - x1) * (y2 - y1)

    keep = []
    ordered = np.arange(len(boxes))
    while ordered.size > 0:
        # Index of the current element:
        i = ordered[0]
        keep.append(boxes[i])
        xx1 = np.maximum(x1[i], x1[ordered[1:]])
        yy1 = np.maximum(y1[i], y1[ordered[1:]])
        xx2 = np.minimum(x2[i], x2[ordered[1:]])
        yy2 = np.minimum(y2[i], y2[ordered[1:]])

        width1 = np.maximum(0.0, xx2 - xx1 + 1)
        height1 = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width1 * height1
        union = (areas[i] + areas[ordered[1:]] - intersection)

        # Compute the Intersection over Union (IoU) score:
        iou = intersection / union

        # only elements whose overlap with the current bounding
        # box is lower than the threshold:
        indexes = np.where(iou <= nms_threshold)[0]
        ordered = ordered[indexes + 1]

    return keep


def add_boxes_to_img(bboxes, img, ground_truth_avail):
    """add bounding boxes detections to img

    Args:
        bboxes: bounding box detections for a single img, if ground_truth_avail
            = True, bboxes is [tp_box, fp_box, fn_box], otherwise, bboxes is
            list of [label, x1, y1, x2, y2]
        img: the img to draw bounding boxes
        ground_truth_avail: whether grounding truth results are avilable, used
            to determine the formats of bboxes

    Returns:
        img: img with bounding boxes drawed
    """
    colors = {'r': (0, 0, 255), 'g': (0, 255, 0), 'y': (0, 255, 255)}
    if ground_truth_avail:
        # green for tp boxes
        for label, x1, y1, x2, y2 in bboxes[0]:
            color = colors['g']
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            _ = cv2.putText(img, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)
        # yellow for fp boxes
        for label, x1, y1, x2, y2 in bboxes[1]:
            color = colors['y']
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            _ = cv2.putText(img, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)
        # red for fn boxes
        for label, x1, y1, x2, y2 in bboxes[2]:
            color = colors['r']
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            _ = cv2.putText(img, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)
    else:
        # green for all detected boxes
        for label, x1, y1, x2, y2 in bboxes:
            color = colors['g']
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            _ = cv2.putText(img, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)
    return img


skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
            (1, 3), (2, 4), (3, 5), (4, 6)]


def add_kp_to_img(pred, img, ground_truth_avail):
    """add human keypoint detections to img

    Args:
        kp: keypoint detection for a single image
        img: a single image

    Returns:
        img: img with keypoints drawed
    """
    colors = {'r': (0, 0, 255), 'g': (0, 255, 0), 'y': (0, 255, 255)}
    height, width, _ = img.shape
    if ground_truth_avail:

        # green for tp kps
        for dic in pred[0]:
            for _, (x, y, _) in dic.items():
                cv2.circle(img, (int(x), int(y)), 3, colors['g'], -1)
            for kp_ind1, kp_ind2 in skeleton:
                if kp_ind1 in dic and kp_ind2 in dic:
                    x1, y1, _ = dic[kp_ind1]
                    x2, y2, _ = dic[kp_ind2]
                    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                             colors['g'], 1)

        # yellow for fp kps
        for dic in pred[1]:
            for _, (x, y, _) in dic.items():
                cv2.circle(img, (int(x), int(y)), 3, colors['y'], -1)
            for kp_ind1, kp_ind2 in skeleton:
                if kp_ind1 in dic and kp_ind2 in dic:
                    x1, y1, _ = dic[kp_ind1]
                    x2, y2, _ = dic[kp_ind2]
                    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                             colors['y'], 1)

        # red for fn kps
        for dic in pred[2]:
            for kp_ind, (x, y, _) in dic.items():
                cv2.circle(img, (int(x), int(y)), 3, colors['r'], -1)
            for kp_ind1, kp_ind2 in skeleton:
                if kp_ind1 in dic and kp_ind2 in dic:
                    x1, y1, _ = dic[kp_ind1]
                    x2, y2, _ = dic[kp_ind2]
                    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                             colors['r'], 1)

    else:
        for i, pred_object in enumerate(pred):
            pred = pred_object.data
            pred_visible = pred[pred[:, 2] > 0]
            xs = pred_visible[:, 0]
            ys = pred_visible[:, 1]
            for x, y in zip(xs, ys):
                x, y = x / 641 * width, y / 369 * height
                cv2.circle(img, ((int)(x), (int)(y)), 4, colors['g'], -1)
            decode_order = [(a, b)
                            for (a, b, c, d) in pred_object.decoding_order]
            for index, (a, b) in enumerate(decode_order):
                if (a + 1, b + 1) in pred_object.skeleton or (
                        b + 1, a + 1) in pred_object.skeleton:
                    x1, y1, _ = pred_object.decoding_order[index][2]
                    x2, y2, _ = pred_object.decoding_order[index][3]
                else:
                    continue
                x1, x2 = x1 / 641 * width, x2 / 641 * width
                y1, y2 = y1 / 369 * height, y2 / 369 * height
                cv2.line(img, ((int)(x1), (int)(y1)), ((int)(x2), (int)(y2)),
                         colors['g'], 2)
    return img


def add_better(ax, x, y, direc, fontsize=16):
    '''add a better arrow in the given axes

    Args:
        ax: the given axes to draw arrow on
        x: x coord in axes percentage, should be in (0, 1)
        y: y coord in axes percentage, should be in (0, 1)
        direc: direction of the arrow, can be one of down, up, left, right,
            tr (top right), tl (top left), br (bottom right), bl (bottom left)
        fontsize: font size of the text

    Usage:
        fig, ax = plt.subplots(figsize=(5, 3), dpi=200)
        for i in range(5):
            add_better(ax, np.random.random(), np.random.random(), 'tr', \
                np.random.randint(5, 30))
        fig.savefig('./test.pdf', format='pdf', bbox_inches='tight')
    '''
    assert 0 < x < 1 and 0 < y < 1, f'invalid x ({x}) or y ({y}) value'
    assert direc in ['up', 'down', 'left', 'right', 'tr', 'tl', 'br',
                     'bl'], ('invalid direc value ({direc})')

    if direc == 'down':
        ax.annotate('',
                    xy=(x, y - 0.05 / 16 * fontsize),
                    xytext=(0, 51 / 16 * fontsize),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    arrowprops=dict(width=fontsize,
                                    headwidth=fontsize * 1.5,
                                    headlength=fontsize,
                                    color='black',
                                    fill=False))
        ax.text(x - 0.009 / 16 * fontsize,
                y,
                'better',
                fontsize=fontsize,
                color='black',
                ha='center',
                va='bottom',
                rotation=270,
                transform=ax.transAxes)
    elif direc == 'up':
        ax.annotate('',
                    xy=(x, y + 0.25 / 16 * fontsize),
                    xytext=(0, -51 / 16 * fontsize),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    arrowprops=dict(width=fontsize,
                                    headwidth=fontsize * 1.5,
                                    headlength=fontsize,
                                    color='black',
                                    fill=False))
        ax.text(x - 0.005 / 16 * fontsize,
                y,
                'better',
                fontsize=fontsize,
                color='black',
                ha='center',
                va='bottom',
                rotation=270,
                transform=ax.transAxes)
    elif direc == 'left':
        ax.annotate('',
                    xy=(x, y),
                    xytext=(51 / 16 * fontsize, 0),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    arrowprops=dict(width=fontsize,
                                    headwidth=fontsize * 1.5,
                                    headlength=fontsize,
                                    color='black',
                                    fill=False))
        ax.text(x + 0.105 / 16 * fontsize,
                y - 0.05 / 16 * fontsize,
                'better',
                fontsize=fontsize,
                color='black',
                ha='center',
                va='bottom',
                rotation=0,
                transform=ax.transAxes)
    elif direc == 'right':
        ax.annotate('',
                    xy=(x, y),
                    xytext=(-51 / 16 * fontsize, 0),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    arrowprops=dict(width=fontsize,
                                    headwidth=fontsize * 1.5,
                                    headlength=fontsize,
                                    color='black',
                                    fill=False))
        ax.text(x - 0.105 / 16 * fontsize,
                y - 0.05 / 16 * fontsize,
                'better',
                fontsize=fontsize,
                color='black',
                ha='center',
                va='bottom',
                rotation=0,
                transform=ax.transAxes)
    elif direc == 'bl':
        ax.annotate('',
                    xy=(x, y),
                    xytext=(36 / 16 * fontsize, 36 / 16 * fontsize),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    arrowprops=dict(width=fontsize,
                                    headwidth=fontsize * 1.5,
                                    headlength=fontsize,
                                    color='black',
                                    fill=False))
        ax.text(x + 0.075 / 16 * fontsize,
                y + 0.012 / 16 * fontsize,
                'better',
                fontsize=fontsize,
                color='black',
                ha='center',
                va='bottom',
                rotation=45,
                transform=ax.transAxes)
    elif direc == 'tl':
        ax.annotate('',
                    xy=(x, y),
                    xytext=(36 / 16 * fontsize, -36 / 16 * fontsize),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    arrowprops=dict(width=fontsize,
                                    headwidth=fontsize * 1.5,
                                    headlength=fontsize,
                                    color='black',
                                    fill=False))
        ax.text(x + 0.07 / 16 * fontsize,
                y - 0.25 / 16 * fontsize,
                'better',
                fontsize=fontsize,
                color='black',
                ha='center',
                va='bottom',
                rotation=315,
                transform=ax.transAxes)
    elif direc == 'tr':
        ax.annotate('',
                    xy=(x, y),
                    xytext=(-36 / 16 * fontsize, -36 / 16 * fontsize),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    arrowprops=dict(width=fontsize,
                                    headwidth=fontsize * 1.5,
                                    headlength=fontsize,
                                    color='black',
                                    fill=False))
        ax.text(x - 0.07 / 16 * fontsize,
                y - 0.24 / 16 * fontsize,
                'better',
                fontsize=fontsize,
                color='black',
                ha='center',
                va='bottom',
                rotation=45,
                transform=ax.transAxes)
    elif direc == 'br':
        ax.annotate('',
                    xy=(x, y),
                    xytext=(-36 / 16 * fontsize, 36 / 16 * fontsize),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    arrowprops=dict(width=fontsize,
                                    headwidth=fontsize * 1.5,
                                    headlength=fontsize,
                                    color='black',
                                    fill=False))
        ax.text(x - 0.075 / 16 * fontsize,
                y - 0.012 / 16 * fontsize,
                'better',
                fontsize=fontsize,
                color='black',
                ha='center',
                va='bottom',
                rotation=315,
                transform=ax.transAxes)
