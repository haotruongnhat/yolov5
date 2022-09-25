from pathlib import Path
from interfaces.socket_chunk import chunk_send, chunk_recv
import json
import socket
import numpy as np
import cv2
from inference_utils import *

import onnxruntime

# def infer(model, image_size, stride, config, max_det=300):
#     dt, seen = [0.0, 0.0, 0.0], 0

#     image_path = config["im_path"]
#     conf_thres = config["detect_config"]["conf_thres"]
#     nms = config["detect_config"]["nms"]
#     eps_enable = config["filter_config"]["eps_enable"]
#     eps_m = config["filter_config"]["eps_m"]
#     eps_o = config["filter_config"]["eps_o"]
#     num_samples = config["filter_config"]["num_samples"]

#     if isinstance(image_path, str):
#         im = cv2.imread(image_path)
#     else:
#         im = image_path

#     if im is None:
#         print("ERROR in reading image")
#         return [dict(boxes=[], scores=[], classes=[])]

#     im0 = im.copy()
#     im = letterbox(im, image_size, stride=stride, auto=False)[0]
#     im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
#     im = np.ascontiguousarray(im)

#     result_list = []

#     t1 = time_sync()
#     # im = torch.from_numpy(im).to(device)
#     im = im.astype(np.float32) # uint8 to fp16/32
#     im /= 255  # 0 - 255 to 0.0 - 1.0
#     if len(im.shape) == 3:
#         im = im[None]  # expand for batch dim
#     t2 = time_sync()
#     dt[0] += t2 - t1

#     pred = model.run([model.get_outputs()[0].name], {model.get_inputs()[0].name: im})[0]
#     t3 = time_sync()
#     dt[1] += t3 - t2

#     pred = non_max_suppression(pred, conf_thres, nms, multi_label=False, max_det=max_det)
#     dt[2] += time_sync() - t3
#     # Process predictions
#     for i, det in enumerate(pred):  # per image
#         seen += 1
#         result_dict = {}
#         result_dict["boxes"] = []
#         result_dict["scores"] = []

#         if len(det):
#             # Rescale boxes from img_size to im0 size
#             det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

#             # Write results
#             boxes = []; scores = []; classes = []
#             for *xyxy, conf, cls in reversed(det):
#                 # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                 confidence = conf.item()
#                 if confidence >= conf_thres:
#                     boxes.append(list(xyxy))
#                     scores.append(confidence)
#                     classes.append(cls.item())

#             if eps_enable:
#                 boxes, scores, classes = group_filter(boxes, scores, classes, eps_m, eps_o, num_samples)

#             result_dict["boxes"] = boxes
#             result_dict["scores"] = scores
#             result_dict["classes"] = classes

#         result_list.append(result_dict)

#     # Print results
#     t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
#     LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS' % t)

#     return result_list

if __name__ == "__main__":
    with open("config.json") as f:
        args = json.load(f)

    imgsz = [int(v) for v in args["size"].split(",")]
    model_path = None
    findall_onnx = list(Path(".").glob("*.onnx"))
    if len(findall_onnx) > 0:
        model_path = str(findall_onnx[0])
    else:
        model_path = args["model_path"]

    # model_path = "data_new_view_retrain_july.onnx"
    weights = model_path

    print("Load ONNX file:", model_path)

    provider = ['CPUExecutionProvider']
    if args["device"] == "gpu":
        provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    model = onnxruntime.InferenceSession(model_path, providers=provider)
    print("Running on", args["device"])
    
    meta = model.get_modelmeta().custom_metadata_map 
    stride, names = int(meta['stride']), eval(meta['names'])

    print("Dummy run. First run take longer time than normal")
    dummy_img = np.zeros((964, 1294, 3))
    # dummy_img = "01042022_2_2.jpg" # Toggle this to dummy as image
    detect_config = dict(conf_thres=0.6, nms=0.25)
    filter_config = dict(eps_enable=True, eps_m=2, eps_o=0, num_samples=3)
    config = dict(im_path = dummy_img, detect_config=detect_config, filter_config=filter_config)

    infer(model, imgsz, stride, config)[0] # Only one image
    print("Dummy run succesfully")

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        ip = "127.0.0.1"
        server_addr = (ip, args["server_port"])
        print("Server address: ", server_addr)

        client_addr = (ip, args["client_port"])
        print("Client address: ", client_addr)

        s.bind(server_addr) 
    
        print("Start loop")
        while True:
            print("Waiting for message")
            config = chunk_recv(s)

            # print("Received data: {}".format(config))
            print("Received data")

            ## Inference
            result_dict = infer(model, imgsz, stride, config)[0] # Only one image
            print("Number of boxes returned:", len(result_dict["boxes"]))

            print("Send result back to: {}".format(client_addr))
            chunk_send(result_dict, s, client_addr)
            print("==============================================")
    except:
        s.close()