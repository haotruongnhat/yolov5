import os
from sklearn.cluster import DBSCAN
import time
import numpy as np
import cv2
import logging 

CLASSES = ["thep",]
VERBOSE = str(os.getenv('VERBOSE', True)).lower() == 'true'  # global verbose mode

def infer(model, image_size, stride, config, max_det=300, verbose=True):
    dt, seen = [0.0, 0.0, 0.0], 0

    image_path = config["im_path"]
    conf_thres = config["detect_config"]["conf_thres"]
    nms = config["detect_config"]["nms"]
    eps_enable = config["filter_config"]["eps_enable"]
    eps_m = config["filter_config"]["eps_m"]
    eps_o = config["filter_config"]["eps_o"]
    num_samples = config["filter_config"]["num_samples"]

    if isinstance(image_path, str):
        im = cv2.imread(image_path)
    else:
        im = image_path

    if im is None:
        print("ERROR in reading image")
        return [dict(boxes=[], scores=[], classes=[])]

    im0 = im.copy()
    im = letterbox(im, image_size, stride=stride, auto=False)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    result_list = []

    t1 = time_sync()
    # im = torch.from_numpy(im).to(device)
    im = im.astype(np.float32) # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1

    pred = model.run([model.get_outputs()[0].name], {model.get_inputs()[0].name: im})[0]
    t3 = time_sync()
    dt[1] += t3 - t2

    pred = non_max_suppression(pred, conf_thres, nms, multi_label=False, max_det=max_det)
    dt[2] += time_sync() - t3
    # Process predictions
    for i, det in enumerate(pred):  # per image
        seen += 1
        result_dict = {}
        result_dict["boxes"] = []
        result_dict["scores"] = []

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            boxes = []; scores = []; classes = []
            for *xyxy, conf, cls in reversed(det):
                # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                confidence = conf.item()
                if confidence >= conf_thres:
                    boxes.append([int(v) for v in list(xyxy)])
                    scores.append(confidence)
                    classes.append(cls.item())

            if eps_enable:
                boxes, scores, classes = group_filter(boxes, scores, classes, eps_m, eps_o, num_samples)

            result_dict["boxes"] = boxes
            result_dict["scores"] = scores
            result_dict["classes"] = classes

        result_list.append(result_dict)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if verbose:
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS' % t)

    return result_list

def set_logging(name=None, verbose=VERBOSE):
    # Sets level and returns logger
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING
    log = logging.getLogger(name)
    log.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(level)
    log.addHandler(handler)


set_logging()  # run before defining LOGGER
LOGGER = logging.getLogger("demthep")  # define globally (used in train.py, val.py, detect.py, etc.)

def time_sync():
    return time.time()

def create_dir(dir):
    try:
        os.makedirs(dir)
    except: 
        pass

def draw(image, result, print_score=True):
    im = cv2.imread(image)
    for box, score in zip(result["boxes"], result["scores"]):
        x1, y1, x2, y2 = box
        im = cv2.circle(im, (int((x1+x2)/2), int((y1+y2)/2)), int(min([x2-x1, y2-y1])/2) - 4, (255, 0, 0), thickness=3)

    for box, score in zip(result["boxes"], result["scores"]):
        score_str = str(round(score, 2))
        x1, y1, x2, y2 = box

        if print_score:
            cv2.putText(im, score_str, (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2)

    cv2.putText(im, str(len(result["boxes"])), (100, 100),  cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), thickness=3)

    return im

def save_txt(path, result, image_shape):
    gn = np.array(image_shape)[[1, 0, 1, 0]]
    with open(path, 'w') as f:
        if result["boxes"]:
            for box, cls in zip(result["boxes"], result["classes"]):
                xyxy = np.array(box)
                xywh = (xyxy2xywh(xyxy.reshape(1, -1)) / gn).reshape(-1) .tolist()  # normalized xywh
                line = cls, *xywh # label format
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

def save_classes_txt(path):
    with open(path, 'a') as f:
        f.write("thep" + '\n')

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3].clip(0, img_shape[0])  # y2

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def nms_numpy(boxes, scores, threshold, method='Min'):
    if boxes.size == 0:
        return np.empty((0, 3))

    x1 = boxes[:, 0].copy()
    y1 = boxes[:, 1].copy()
    x2 = boxes[:, 2].copy()
    y2 = boxes[:, 3].copy()
    s = scores
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]

        xx1 = np.maximum(x1[i], x1[idx]).copy()
        yy1 = np.maximum(y1[i], y1[idx]).copy()
        xx2 = np.minimum(x2[i], x2[idx]).copy()
        yy2 = np.minimum(y2[i], y2[idx]).copy()

        w = np.maximum(0.0, xx2 - xx1 + 1).copy()
        h = np.maximum(0.0, yy2 - yy1 + 1).copy()

        inter = w * h
        if method == 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= threshold)]

    pick = pick[:counter].copy()
    return pick

def non_max_suppression(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        classes=None,
                        agnostic=False,
                        multi_label=False,
                        labels=(),
                        max_det=300):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.1 + 0.03 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [np.zeros((0, 6))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros((len(lb), nc + 5))
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = np.concatenate((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero()
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].astype(np.float32)), 1)
        else:  # best class only
            conf = x[:, 5:].max(1, keepdims=True)
            j = np.zeros(conf.shape).astype(np.float32)
            x = np.concatenate((box, conf, j), 1)[conf.reshape(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms_numpy(boxes, scores, iou_thres)
        # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = np.dot(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output

def group_filter(bboxes, scores, classes, eps_m, eps_o, num_samples):
    if not bboxes:
        return [], []

    bboxes_np = np.array(bboxes)
    # import pdb; pdb.set_trace()
    x = (bboxes_np[:, 2] + bboxes_np[:, 0])/2
    y = (bboxes_np[:, 3] + bboxes_np[:, 1])/2
    bboxes_center = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), 1) 

    average_size = np.mean(bboxes_np[:, 2] - bboxes_np[:, 0])

    eps_coeff = eps_m*average_size + eps_o

    db = DBSCAN(eps=eps_coeff, min_samples=num_samples).fit(bboxes_center)
    # print(db.labels_)
    unq, counts = np.unique(db.labels_, return_counts=True)
    largest_blob_index = np.argmax(counts)
    largest_index =  unq[largest_blob_index]

    selected_indexes = np.where( np.array(db.labels_) == largest_index )
    filter_bboxes = np.array(bboxes)[selected_indexes]
    filter_scores = np.array(scores)[selected_indexes]
    filter_classes = np.array(classes)[selected_indexes]

    return filter_bboxes.tolist(), filter_scores.tolist(), filter_classes.tolist()