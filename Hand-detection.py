import mxnet as mx
import numpy as np
from collections import namedtuple
import cv2

Batch = namedtuple('Batch', ['data'])
video = cv2.VideoCapture(0) 
ok, img = video.read()
short = 300    #you can change short size what you need
mult_base = 1
interp = 2
max_size = 1024
h, w, _ = img.shape
im_size_min, im_size_max = (h, w) if w > h else (w, h)
scale = float(short) / float(im_size_min)
if np.round(scale * im_size_max / mult_base) * mult_base > max_size:
    # fit in max_size
    scale = float(np.floor(max_size / mult_base) * mult_base) / float(im_size_max)
new_w, new_h = (int(np.round(w * scale / mult_base) * mult_base),
                int(np.round(h * scale / mult_base) * mult_base))

net, arg_params, aux_params = mx.model.load_checkpoint('.\ssd_512_mobilenet1.0_custom', 0)

mod = mx.mod.Module(net, label_names = ['data'], context = mx.gpu(0))
mod.bind(data_shapes = [('data', (1, 3, new_h, new_w))])
mod.set_params(arg_params, aux_params)

while True:
    ok, img = video.read()
    if not ok:
        break
    img = cv2.resize(img, (new_w, new_h), interpolation = interp)
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32')
    mean = 255 * np.array([0.485, 0.456, 0.406])
    dst = (RGB - mean) / np.array([0.229, 0.224, 0.225])
    # whc chw
    dst = np.swapaxes(dst, 0, 2)
    dst = np.swapaxes(dst, 1, 2)
    dst = dst[np.newaxis, :]
    db = Batch([mx.nd.array(dst, ctx = mx.gpu(0))])
    mod.forward(db)
    cls_id = mod.get_outputs()[0][0].asnumpy()
    scores = mod.get_outputs()[1][0].asnumpy()
    bbox = mod.get_outputs()[2].asnumpy()
    scores = scores.reshape(100)
    keep = np.where(scores > 0.5)
    bboxk = bbox.reshape(100, 4)[keep]
    scoresk = scores[keep]
    for i in range(len(bboxk)):
        xmin, ymin, xmax, ymax = bboxk[i]
        img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
        cv2.putText(img, str(scoresk[i]), (int(xmin), int(ymin)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
    cv2.imshow("", img)
    cv2.waitKey(1)
