#!/usr/bin/python
import socket
import pickle
import json
import codecs

LABVIEW_MODE = True

def check_input_parameters(total_samples, conf_thres, nms, \
                                eps_enable, eps_m, eps_o, eps_samples, \
                                    black_filter_enable, black_filter_threshold):
    if (total_samples < 0):
        raise Exception("`total_samples` should be greater than 0: {}".format(total_samples))

    if (conf_thres < 0.0) or (conf_thres > 1.0):
        raise Exception("`conf_thres` should be between 0.0 and 1.0. Input: {}".format(conf_thres))

    if (nms < 0.0) or (nms > 1.0):
        raise Exception("`nms` should be between 0.0 and 1.0. Input: {}".format(nms))

    if not((eps_enable == True) or (eps_enable == False)):
        raise Exception("`eps_enable` should be True or False: {}".format(eps_enable))

    if (eps_m < 0):
        raise Exception("`eps_m` should be greater than 0: {}".format(eps_m))

    if (eps_o < 0):
        raise Exception("`eps_o` should be greater than 0: {}".format(eps_o))

    if (eps_samples < 0):
        raise Exception("`eps_samples` should be greater than 0: {}".format(eps_samples))

    if not((black_filter_enable == True) or (black_filter_enable == False)):
        raise Exception("`black_filter_enable` should be True or False: {}".format(black_filter_enable))

    if (black_filter_threshold < 0) or (black_filter_threshold > 255):
        raise Exception("`black_filter_threshold` should be between 0 and 255. Input: {}".format(black_filter_threshold))

def send_config(im_path, total_samples, \
                conf_thres=0.6, nms=0.5, \
                    eps_enable=True, eps_m=4, eps_o= 0, eps_samples=3, \
                        black_filter_enable=True, black_filter_threshold=80):

    eps_enable = json.loads(eps_enable.lower())
    black_filter_enable = json.loads(black_filter_enable.lower())

    check_input_parameters(total_samples = total_samples, \
                            conf_thres = conf_thres, nms = nms, \
                                eps_enable= eps_enable, eps_m= eps_m, eps_o=  eps_o, eps_samples= eps_samples, \
                                    black_filter_enable= black_filter_enable, black_filter_threshold = black_filter_threshold)

    detect_config = dict(conf_thres=conf_thres, nms=nms)
    filter_config = dict(eps_enable=eps_enable, eps_m=eps_m, eps_o=eps_o, eps_samples=eps_samples,\
                        black_sample_filter_enable=black_filter_enable, black_sample_threshold=black_filter_threshold)
    config = dict(im_path = im_path, total_samples=total_samples, detect_config=detect_config, filter_config=filter_config)

    return json.dumps(config)

def receive_result(received_str):
    result_dict = json.loads(received_str)

    processed_dict = {}
    processed_dict["boxes"] = []; processed_dict["classes"] = []
    class_name = "thep"
    if len(result_dict["boxes"]):
        for bbox, score, cls in zip(result_dict["boxes"], result_dict["scores"], result_dict["classes"]):
            processed_dict["boxes"].append(
                {
                    str(score) : [int(v) for v in bbox]
                }
            )
            processed_dict["classes"].append(cls)

    return json.dumps(processed_dict)

if __name__ == "__main__":
    ip = "127.0.0.1"
    server_addr = 50000
    client_addr = 50004

    image_path = "D:\Projects\VSTech\yolov5\outputs\data_september_error_for_heuristics\Sun, Sep 25, 2022_2_199.jpg"
    
    send_str = send_config(image_path, \
                                total_samples=150, conf_thres=0.6, nms=0.5, \
                                    eps_enable="true", eps_m=4, eps_o= 0, eps_samples=3, \
                                        black_filter_enable="true", black_filter_threshold=80)

    ### Labview UDP send
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", int(client_addr)))

        s.sendto(codecs.encode(send_str, "utf-8"), (ip, int(server_addr)))

    ### Labview UDP Receive
        receive_str = s.recv(4096000).decode("utf-8")

    print(receive_result(receive_str))
