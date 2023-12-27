from typing import List, AnyStr, Tuple, Union,Optional
import numpy as np
import cv2
import os.path as osp
import os
import time
import argparse


class Detector:

    def __init__(self, model:List or AnyStr or Tuple):
        if isinstance(model, list) or '.param' in model or '.bin' in model:
            try:
                import ncnn
            except ImportError:
                raise ImportError(
                    'You have not installed ncnn yet, please execute the "pip install ncnn" command to install and run again'
                )
            net = ncnn.Net()
            if isinstance(model, str):
                if '.bin' in model:
                    model = [model, model.replace('.bin', '.param')]
                else:
                    model = [model, model.replace('.param', '.bin')]
            for p in model:
                if p.endswith('param'):
                    param = p
                if p.endswith('bin'):
                    bin = p
            # load model weights
            net.load_param(param)
            net.load_model(bin)
            net.opt.use_vulkan_compute = False
            self.engine = 'ncnn'
            self._input_shape = [3, 640, 640]
        elif model.endswith('onnx'):
            try:
                import onnxruntime
            except ImportError:
                raise ImportError(
                    'You have not installed onnxruntime yet, please execute the "pip install onnxruntime" command to install and run again'
                )
            try:
                net = onnx.load(model)
                onnx.checker.check_model(net)
            except Exception:
                raise ValueError(
                    'onnx file have error,please check your onnx export code!')
            providers = (
                # ['CUDAExecutionProvider', 'CPUExecutionProvider']
                # if torch.cuda.is_available()
                ['CPUExecutionProvider'])
            net = onnxruntime.InferenceSession(model, providers=providers)

            self._input_shape = net.get_inputs()[0].shape[1:]
            channels = self._input_shape.pop(0)
            self._input_type = net.get_inputs()[0].type
            self._input_shape.append(channels)
            self.engine = 'onnx'
        elif model.endswith('tflite'):
            try:
                import tflite_runtime.interpreter as tflite
            except ImportError:
                raise ImportError(
                    'You have not installed tensorflow yet, please execute the "pip install tensorflow" command to install and run again'
                )
            inter = tflite.Interpreter
            net = inter(model)
            self._input_shape = tuple(net.get_input_details()[0]['shape'][1:])
            net.allocate_tensors()
            self.engine = 'tf'
        else:
            raise 'model file input error'
        self.inter = net

    @property
    def input_shape(self):
        return self._input_shape

    def __call__(
        self,
        img: np.array,
        input_name: AnyStr = 'input',
        output_name: AnyStr = 'output',

        result_num=1,
    ):
        
        if len(img.shape) == 2:  # audio
            if img.shape[1] > 10:  # (1, 8192) to (8192, 1)
                img = img.transpose(1, 0) if self.engine == 'tf' else img
            img = np.array([img])  # add batch dim.
        elif len(img.shape) == 3:
            C, H, W = img.shape
            if C not in [1, 3]:
                img = img.transpose(2, 0, 1)
            # if isinstance(img, torch.Tensor):
            # img = img.numpy()
            img = np.array([img])  # add batch dim.
        elif len(img.shape) == 4:
            B, C, H, W = img.shape
            if C not in [1, 3]:
                img = img.transpose(0, 3, 1, 2)
            # if isinstance(img, torch.Tensor):
            # img = img.numpy()

        else:  # error
            raise ValueError
        results = []
        if self.engine == 'onnx':  # onnx
            result = self.inter.run([
                self.inter.get_outputs()[0].name,
                self.inter.get_outputs()[1].name
            ], {self.inter.get_inputs()[0].name: img})
            results = result
            # results.append(result[0])

        elif self.engine == 'ncnn':  # ncnn
            import ncnn

            self.inter.opt.use_vulkan_compute = False
            extra = self.inter.create_extractor()
            input_name = self.inter.input_names()[0]
            output_name = self.inter.output_names()[0]
            extra.input(input_name, ncnn.Mat(img[0]))  # noqa
            result = extra.extract(output_name)[1]
            return [np.expand_dims(np.array(result), axis=0)]
        else:  # tf
            input_, outputs = self.inter.get_input_details()[0], (
                self.inter.get_output_details()[i] for i in range(result_num))
            int8 = input_['dtype'] == np.int8 or input_['dtype'] == np.uint8
            img = img.transpose(0, 2, 3, 1) if len(img.shape) == 4 else img
            if int8:
                scale, zero_point = input_['quantization']
                img = (img / scale + zero_point).astype(np.int8)
            self.inter.set_tensor(input_['index'], img)
            self.inter.invoke()
            for output in outputs:
                result = self.inter.get_tensor(output['index'])
                if int8:
                    scale, zero_point = output['quantization']
                    result = (result.astype(np.float32) - zero_point) * scale
                results.append(result)

        return results

IMG_SUFFIX = ('.jpg', '.png', '.PNG', '.jpeg')
VIDEO_SUFFIX = ('.avi', '.mp4', '.mkv', '.flv', '.wmv', '.3gp')
IOT_DEVICE = ('sensorcap',)

class DataStream:
    def __init__(self, source: Union[int, str], shape: Optional[int or Tuple[int, int]] = None) -> None:
        if shape:
            self.gray = True if shape[-1] == 1 else False
            self.shape = shape[:-1]
        else:
            self.gray = False
            self.shape = shape
        self.file = None
        self.l = 0

        if isinstance(source, str):
            if osp.isdir(source):
                self.file = [osp.join(source, f) for f in os.listdir(source) if f.lower().endswith(IMG_SUFFIX)]
                self.l = len(self.file)
                self.file = iter(self.file)

            elif osp.isfile(source):
                if any([source.lower().endswith(mat) for mat in IMG_SUFFIX]):
                    self.file = [source]
                    self.l = len(self.file)
                    self.file = iter(self.file)
                elif any([source.lower().endswith(mat) for mat in VIDEO_SUFFIX]):
                    self.cap = cv2.VideoCapture(source)
            elif source.isdigit():
                self.cap = cv2.VideoCapture(int(source))
            # elif source in IOT_DEVICE:
            # self.cap = IoTCamera()
            else:
                raise
        elif isinstance(source, int):
            self.cap = cv2.VideoCapture(source)
        else:
            raise

    def __len__(self):
        return self.l if self.file else None

    def __iter__(self):
        return self

    def __next__(self):
        if self.file:
            f = next(self.file)
            # img = load_image(f, shape=self.shape, mode='GRAY' if self.gray else 'RGB', normalized=True)

        else:
            while True:
                ret, raw = self.cap.read()
                img = raw
                if ret:
                    break

            if self.gray:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = np.expand_dims(img, axis=-1)

            if self.shape:
                img = cv2.resize(img, self.shape[::-1])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img / 255).astype(np.float32)

        return raw, img

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def xyxy2cocoxywh(x, coco_format: bool = False):
    y = np.copy(x)
    # top left x or center x
    y[:, 0] = x[:, 0] if coco_format else (x[:, 0] + x[:, 2]) / 2
    # top left y or center y
    y[:, 1] = x[:, 1] if coco_format else (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def NMS(
    bbox: np.ndarray,
    confiden: np.ndarray,
    classer: np.ndarray,
    bbox_format='xyxy',
    max_det=300,
    iou_thres=0.4,
    conf_thres=25,
):
    # bbox = bbox if isinstance(bbox, torch.Tensor) else torch.from_numpy(bbox)
    # confiden = confiden if isinstance(confiden, torch.Tensor) else torch.from_numpy(confiden)
    # classer = classer if isinstance(classer, torch.Tensor) else torch.from_numpy(classer)

    assert bbox.shape[0] == confiden.shape[0] == classer.shape[0]

    conf_mask = confiden[0:] > conf_thres

    confiden = confiden[conf_mask]
    bbox = bbox[conf_mask]
    classer = classer[conf_mask]
    
    

    if bbox_format == 'xyxy':
        pass
    elif bbox_format == 'xywh':
        bbox = xywh2xyxy(bbox)

    pred = np.concatenate((bbox, confiden.reshape(-1, 1), np.expand_dims(np.argmax(classer, axis=1), axis=1)), axis=1)

    if pred.shape[0] > 0:
        pred = pred[pred[:, 4].argsort(axis=0)[::-1][:max_det]]

    bbox, confiden = pred[:, :4], pred[:, 4]
    return pred[0:1,:]


def show_det(
    pred: np.ndarray,
    img: Optional[np.ndarray] = None,
    img_file: Optional[str] = None,
    win_name='Detection',
    class_name=None,
    shape=None,
    save_path=False,
    show=False,
    fps=None,
) -> np.ndarray:
    assert not (img is None and img_file
                is None), 'The img and img_file parameters cannot both be None'

    # load image
    # else:
    # img = load_image(img_file, shape=shape, mode='BGR').copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    # plot the result
    for i in pred:
        x1, y1, x2, y2 = map(int, i[:4])
        x1 = int(x1 * (img.shape[1] / shape[1]))
        x2 = int(x2 * (img.shape[1] / shape[1]))
        y1 = int(y1 * (img.shape[0] / shape[0]))
        y2 = int(y2 * (img.shape[0] / shape[0]))
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            img,
            class_name[int(i[5])] if class_name else 'None',
            (x1, y1),
            font,
            color=(255, 0, 0),
            thickness=1,
            fontScale=0.5,
        )
        cv2.putText(img,
                    str(round(i[4].item(), 2)), (x1, y1 - 15),
                    font,
                    color=(0, 0, 255),
                    thickness=1,
                    fontScale=0.5)
    if fps is not None:
        cv2.putText(img,
                    f"FPS:{1/fps:.5} Take: {fps*1000:.5} ms", (0, 15),
                    font,
                    color=(0, 0, 255),
                    thickness=1,
                    fontScale=0.6)

    if show:
        cv2.imshow(win_name, img)
        cv2.waitKey(1)

    if save_path:
        img_name = osp.basename(img_file)
        cv2.imwrite(osp.join(save_path, img_name), img)

    return pred


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--model',default='./yolo.tflite')
    args.add_argument('--source', default=0)
    return args.parse_args()

if __name__ == '__main__':
    args=get_args()
    Inference = Detector(args.model)
    Dataloader = DataStream(args.source, Inference.input_shape)
    for raw, data in Dataloader:
        t0 = time.time()
        result = Inference(data)
        t = time.time() - t0
        preds = result[0][0]
        bbox, conf, classes = preds[:, :4], preds[:, 4], preds[:, 5:]
        preds = NMS(bbox, conf, classes, conf_thres=20, bbox_format='xywh')
        show_det(pred=preds,
                 img=raw,
                 class_name=['face'],
                 show=True,
                 shape=Inference.input_shape,
                 fps=t)
