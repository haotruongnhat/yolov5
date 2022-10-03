from pathlib import Path
from interfaces.socket_chunk import chunk_send, chunk_recv
import json
import socket
import numpy as np
import cv2
from inference_utils import *

import onnxruntime

import os
import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <6}</level> | <level>{message}</level>', level="INFO")
logger.add("demthep_detect_{time:YYYY_MM_DD}.log", format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <6}</level> | <level>{message}</level>', level="DEBUG")

if __name__ == "__main__":
    try:
        if not os.path.isfile("config.json"):
            logger.error("Config file not found: config.json")

        with open("config.json") as f:
            args = json.load(f)

        imgsz = [int(v) for v in args["size"].split(",")]
        model_path = None
        findall_onnx = list(Path(".").glob("*.onnx"))
        if len(findall_onnx) > 0:
            model_path = str(findall_onnx[0])
        else:
            model_path = args["model_path"]

        weights = model_path

        if not os.path.isfile(model_path):
            logger.error("Model file not found: ".format(model_path))

        logger.info("Load ONNX file: {}".format(model_path))

        provider = ['CPUExecutionProvider']
        if args["device"] == "gpu":
            provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        model = onnxruntime.InferenceSession(model_path, providers=provider)
        logger.info("Running on: {}".format(args["device"]))
        
        meta = model.get_modelmeta().custom_metadata_map 
        stride, names = int(meta['stride']), eval(meta['names'])

        logger.info("Dummy run. First run take longer time than normal")
        dummy_img = np.zeros((964, 1294, 3))
        detect_config = dict(conf_thres=0.6, nms=0.5)
        filter_config = dict(eps_enable=True, eps_m=4, eps_o=0, eps_samples=3,\
                            black_sample_filter_enable=True, black_sample_threshold=80)
        config = dict(im_path = dummy_img, total_samples=150, detect_config=detect_config, filter_config=filter_config)

        infer(model, imgsz, stride, config)[0] # Only one image
        logger.info("Dummy run succesfully")

        if not ("server_port" in args):
            logger.error("Key not found in config: `server_port`")
        
        if not ("client_port" in args):
            logger.error("Key not found in config: `client_port`")

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        ip = "127.0.0.1"
        server_addr = (ip, args["server_port"])
        logger.info("Server address: {}".format(server_addr))

        client_addr = (ip, args["client_port"])
        logger.info("Client address: {}".format(client_addr))

        s.bind(server_addr) 

        logger.info("Start loop")
        while True:
            logger.info("Waiting for message")
            config = chunk_recv(s)

            # logger.info("Received data: {}".format(config))
            logger.info("Received data")

            ## Inference
            logger.debug("Received config: {}".format(config))
            result_dict = infer(model, imgsz, stride, config)[0] # Only one image
            logger.info("Number of boxes returned: {}".format(len(result_dict["boxes"])))

            logger.info("Send result back to: {}".format(client_addr))
            chunk_send(result_dict, s, client_addr)
            logger.info("==============================================")

    except Exception as e:
        logger.exception(e)
        s.close()
