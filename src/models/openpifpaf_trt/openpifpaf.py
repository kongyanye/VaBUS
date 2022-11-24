import random
import torch
import numpy as np
import cv2
import threading
import time
import queue

import pycuda.driver as cuda
import PIL
import tensorrt as trt

from .trt_decoder import CifCafDecoder

from .datasets.collate import collate_images_anns_meta
from .datasets.image_list import PilImageList


def allocate_buffers(engine):
    host_inputs = []
    cuda_inputs = []
    host_outputs = []
    cuda_outputs = []
    bindings = []
    for i in range(engine.num_bindings):
        binding = engine[i]
        size = trt.volume(
            engine.get_binding_shape(binding)) * engine.max_batch_size
        dims = engine.get_binding_shape(binding)
        if dims[0] < 0:
            size *= -1
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        #host_mem = cuda.pagelocked_empty(size, np.float32)
        host_mem = cuda.pagelocked_empty(size, dtype)

        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(cuda_mem))
        if engine.binding_is_input(binding):
            host_inputs.append(host_mem)
            cuda_inputs.append(cuda_mem)
        else:
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)
    stream = cuda.Stream()  # create a CUDA stream to run inference
    return bindings, host_inputs, cuda_inputs, host_outputs, cuda_outputs, stream


class Openpifpaf:
    def __init__(self, trt_path='models/openpifpaf_trt/engine.trt'):
        self.width, self.height = None, None

        cuda.init()

        self.engine = self._load_engine(trt_path)
        c_device = cuda.Device(torch.cuda.current_device())

        self.cuda_context = c_device.make_context()
        self.engine_context = self.engine.create_execution_context()

        self.bindings, self.host_inputs, self.cuda_inputs, self.host_outputs, self.cuda_outputs, self.stream = allocate_buffers(
            self.engine)

        self.decoder = CifCafDecoder()
        self.warm_up()

        # async infer and decode in batch model
        self.fields_queue = queue.Queue()
        self.result_queue = queue.Queue()
        job = threading.Thread(target=self.async_decode, args=())
        job.start()
        #job.join()

    def _load_engine(self, filepath):
        G_LOGGER = trt.Logger(trt.Logger.INFO)
        with open(filepath, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            return engine

    def infer(self, img, decode=True):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.width is None:
            self.height, self.width = img.shape[:2]

        img_normalized = np.zeros((self.width, self.height))
        img_normalized = cv2.normalize(img, img_normalized, 0, 255,
                                       cv2.NORM_MINMAX)
        image = img_normalized
        image = cv2.resize(image, (641, 369))
        pil_im = PIL.Image.fromarray(image)
        preprocess = None

        data = PilImageList([pil_im], preprocess=preprocess)
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_images_anns_meta)

        for images_batch1, _, __ in loader:
            np_img = images_batch1.numpy()

        bindings = self.bindings
        host_inputs = self.host_inputs
        host_outputs = self.host_outputs
        cuda_inputs = self.cuda_inputs
        cuda_outputs = self.cuda_outputs
        stream = self.stream

        host_inputs[0] = np.ravel(np.zeros_like(np_img))

        self.cuda_context.push()

        np.copyto(host_inputs[0], np.ravel(np_img))
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)

        self.engine_context.execute_async_v2(bindings=bindings,
                                             stream_handle=stream.handle)

        cif = [None] * 1
        caf = [None] * 1
        cif_names = ['cif']
        caf_names = ['caf']
        for i in range(1, self.engine.num_bindings):
            cuda.memcpy_dtoh_async(host_outputs[i - 1], cuda_outputs[i - 1],
                                   stream)

        stream.synchronize()

        for i in range(1, self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            name = self.engine.get_binding_name(i)
            total_shape = np.prod(shape)
            output = host_outputs[i - 1][0:total_shape]
            output = np.reshape(output, tuple(shape))
            if name in cif_names:
                index_n = cif_names.index(name)
                tmp = torch.from_numpy(output[0])
                cif = tmp.cpu().numpy()
            elif name in caf_names:
                index_n = caf_names.index(name)
                tmp = torch.from_numpy(output[0])
                caf = tmp.cpu().numpy()
        heads = [cif, caf]

        self.cuda_context.pop()

        fields = heads

        if decode:
            predictions = self.decoder.decode(fields)
            return predictions
        else:
            return fields

    def infer_batch(self, imgs):
        for img in imgs:
            fields = self.infer(img, decode=False)
            self.fields_queue.put(fields)

        predictions = []
        for _ in range(len(imgs)):
            pred = self.result_queue.get()
            predictions.append(pred)

        return predictions

    def async_decode(self):
        for fields in iter(self.fields_queue.get, None):
            pred = self.decoder.decode(fields)
            self.result_queue.put(pred)

    def __del__(self):
        self.cuda_context.pop()

        del self.cuda_context
        del self.engine_context
        del self.engine

    def save_img(self, img, predictions, output_path):
        img_vis = cv2.resize(img, (641, 369))

        for i, pred_object in enumerate(predictions):
            pred = pred_object.data
            pred_visible = pred[pred[:, 2] > 0]
            xs = pred_visible[:, 0]
            ys = pred_visible[:, 1]
            color = (random.randint(60, 200), random.randint(0, 255),
                     random.randint(0, 255))
            for x, y in zip(xs, ys):
                cv2.circle(img_vis, ((int)(x), (int)(y)), 2, color, -1)
            decode_order = [(a, b)
                            for (a, b, c, d) in pred_object.decoding_order]
            for index, (a, b) in enumerate(decode_order):
                if (a + 1, b + 1) in pred_object.skeleton or (
                        b + 1, a + 1) in pred_object.skeleton:
                    x1, y1, _ = pred_object.decoding_order[index][2]
                    x2, y2, _ = pred_object.decoding_order[index][3]
                else:
                    continue
                cv2.line(img_vis, ((int)(x1), (int)(y1)),
                         ((int)(x2), (int)(y2)), color, 1)

        output_img = img_vis
        output_img = cv2.resize(output_img, (self.width, self.height))
        cv2.imwrite(output_path, output_img)
        return

    def warm_up(self):
        img = cv2.imread('./models/openpifpaf_trt/images/test.jpg')
        _ = self.infer(img)
