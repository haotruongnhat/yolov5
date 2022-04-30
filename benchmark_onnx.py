from pathlib import Path
from unittest import result
from interfaces.socket_chunk import chunk_send, chunk_recv
import numpy as np
from inference_utils import *

import onnxruntime
import argparse

from tqdm import tqdm
import json
import multiprocessing as mp

class InferenceSession(mp.Process):
    def __init__(self, w, provider, opt):
        super(InferenceSession, self).__init__()
        self.w = w
        self.provider = provider
        opt = opt
    
    def run(self):
        model_stem = Path(self.w).stem
            
        model = onnxruntime.InferenceSession(self.w, providers=self.provider)
        
        detect_config = dict(conf_thres=0.4, nms=0.4)
        filter_config = dict(eps_enable=True, eps_m=3, eps_o=0, num_samples=3)
        config = dict(im_path = None, detect_config=detect_config, filter_config=filter_config)

        meta = model.get_modelmeta().custom_metadata_map 
        stride, names = int(meta['stride']), eval(meta['names'])
                
        images = list(Path(opt.data_dir).glob("**\*.jpg"))

        ### Create folder:
        output_overlay_path = os.path.join(opt.output_dir, model_stem, "overlays")
        output_label_path = os.path.join(opt.output_dir, model_stem, "labels")
        output_original_image_path = os.path.join(opt.output_dir, model_stem, "images")
        create_dir(opt.output_dir)
        create_dir(output_overlay_path)
        create_dir(output_label_path)
        create_dir(output_original_image_path)

        ###
        dt, seen = 0.0, 0

        output = {}
        output["time"] = {}

        for image_path in tqdm(images):
            image_name = image_path.name
            image_stem = image_path.stem
            str_im_path = str(image_path)
            config["im_path"] = str(image_path)

            t1 = time_sync()
            result_dict = infer(model, opt.imgsz, stride, config, verbose=False)[0] # Only one image

            if opt.save_label:
                path = os.path.join(output_label_path, image_stem + ".txt")
                save_txt(path, result_dict, opt.imgsz)

            if opt.save_overlay:
                path = os.path.join(output_overlay_path, image_name)
                cv2.imwrite(path, draw(path, result_dict))

            if opt.save_image:
                path = os.path.join(output_original_image_path, image_name)
                cv2.imwrite(path, cv2.imread(str(image_path)))

            if image_path not in output.keys():
                output[str_im_path] = {}

            # output[image_path][model_stem] = {}
            output[str_im_path][model_stem] = result_dict

            t2 = time_sync()

            dt += t2-t1
            seen += 1
        
        t = dt/seen
        
        print("Execution time on average [{}]: {}".format(model_stem, round(t, 2)))
        output["time"][model_stem] = t

        with open(os.path.join(opt.output_dir, model_stem, "inference_data.json"), "w") as f:
            json.dump(output, f)

def inference(w, provider, opt):
    model_stem = Path(w).stem
        
    model = onnxruntime.InferenceSession(w, providers=provider)
    
    detect_config = dict(conf_thres=0.4, nms=0.4)
    filter_config = dict(eps_enable=True, eps_m=3, eps_o=0, num_samples=3)
    config = dict(im_path = None, detect_config=detect_config, filter_config=filter_config)

    meta = model.get_modelmeta().custom_metadata_map 
    stride, names = int(meta['stride']), eval(meta['names'])
            
    images = list(Path(opt.data_dir).glob("**\*.jpg"))

    ### Create folder:
    output_overlay_path = os.path.join(opt.output_dir, model_stem, "overlays")
    output_label_path = os.path.join(opt.output_dir, model_stem, "labels")
    output_original_image_path = os.path.join(opt.output_dir, model_stem, "images")
    create_dir(opt.output_dir)
    create_dir(output_overlay_path)
    create_dir(output_label_path)
    create_dir(output_original_image_path)

    output = {}
    output["time"] = {}

    ###
    dt, seen = 0.0, 0

    for image_path in tqdm(images):
        image_name = image_path.name
        image_stem = image_path.stem
        str_im_path = str(image_path)
        config["im_path"] = str(image_path)

        t1 = time_sync()
        result_dict = infer(model, opt.imgsz, stride, config, verbose=False)[0] # Only one image

        if opt.save_label:
            path = os.path.join(output_label_path, image_stem + ".txt")
            save_txt(path, result_dict, opt.imgsz)

        if opt.save_overlay:
            path = os.path.join(output_overlay_path, image_name)
            cv2.imwrite(path, draw(path, result_dict))

        if opt.save_image:
            path = os.path.join(output_original_image_path, image_name)
            cv2.imwrite(path, cv2.imread(str(image_path)))

        if image_path not in output.keys():
            output[str_im_path] = {}

        # output[image_path][model_stem] = {}
        output[str_im_path][model_stem] = result_dict

        t2 = time_sync()

        dt += t2-t1
        seen += 1
    
    t = dt/seen
    
    print("Execution time on average [{}]: {}".format(model_stem, round(t, 2)))
    output["time"][model_stem] = t

    with open(os.path.join(opt.output_dir, model_stem, "inference_data.json"), "w") as f:
        json.dump(output, f)
    return output

def merge(a, b, path=None):
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

def get_max_iou(pred_boxes, gt_box):
    """
    calculate the iou multiple pred_boxes and 1 gt_box (the same one)
    pred_boxes: multiple predict  boxes coordinate
    gt_box: ground truth bounding  box coordinate
    return: the max overlaps about pred_boxes and gt_box
    """
    # 1. calculate the inters coordinate
    if pred_boxes.shape[0] > 0:
        ixmin = np.maximum(pred_boxes[:, 0], gt_box[0])
        ixmax = np.minimum(pred_boxes[:, 2], gt_box[2])
        iymin = np.maximum(pred_boxes[:, 1], gt_box[1])
        iymax = np.minimum(pred_boxes[:, 3], gt_box[3])

        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)

    # 2.calculate the area of inters
        inters = iw * ih

    # 3.calculate the area of union
        uni = ((pred_boxes[:, 2] - pred_boxes[:, 0] + 1.) * (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.) +
               (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
               inters)

    # 4.calculate the overlaps and find the max overlap ,the max overlaps index for pred_box
        iou = inters / uni
        iou_max = np.max(iou)
        nmax = np.argmax(iou)
        return iou, iou_max, nmax
    else:
        return None, 0.0, 0

def get_diff_boxes(gt_boxes, pred_boxes, iou_thres = 0.7):
    more_boxes = []
    for box in pred_boxes:
        iou, iou_max, nmax = get_max_iou(gt_boxes, box)

        if iou_max < iou_thres:
            ## Save to false box to draw
            more_boxes.append(box.tolist())

    return more_boxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--weights', nargs='+', type=str)
    parser.add_argument('--imgsz', '--img', '--img-size', type=str, default="960,1280", help='image (h, w)')
    parser.add_argument('--save_label', action='store_true')
    parser.add_argument('--save_image', action='store_true')
    parser.add_argument('--save_overlay', action='store_true')

    parser.add_argument('--save_difference', action='store_true')

    opt = parser.parse_args()
    opt.imgsz = [int(v) for v in opt.imgsz.split(",")]

    provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    dummy_img = np.zeros((964, 1294, 3))
    detect_config = dict(conf_thres=0.4, nms=0.4)
    filter_config = dict(eps_enable=True, eps_m=3, eps_o=0, num_samples=3)
    config = dict(im_path = dummy_img, detect_config=detect_config, filter_config=filter_config)

    # mp.set_start_method('spawn')
    # sess1 = mp.Process(target = inference, args=(opt.weights[0], provider, opt, ))#InferenceSession(opt.weights[0], provider, opt)
    # sess2 = mp.Process(target = inference, args=(opt.weights[1], provider, opt, )) #InferenceSession(opt.weights[1], provider, opt)
    # sess1.start(); sess2.start()
    # sess1.join(); sess2.join()

    # output1 = inference(opt.weights[0], provider, opt)
    # output2 = inference(opt.weights[1], provider, opt)

    with open(os.path.join(opt.output_dir, Path(opt.weights[0]).stem, "inference_data.json")) as f:
        output1 = json.load(f)
    with open(os.path.join(opt.output_dir, Path(opt.weights[1]).stem, "inference_data.json")) as f:
        output2 = json.load(f)
    
    output = merge(output1, output2)

    ## Comparision on two models
    if len(opt.weights) == 2:
        output_diff_dir = os.path.join(opt.output_dir, "diff")
        create_dir(output_diff_dir)

        output_meta_file = os.path.join(output_diff_dir, "metadata.txt")
        meta_file = open(output_meta_file, "w")

        count_diff = 0
        print("Compare two models' results")
        # Write output image that differs from both
        for index, image_path in tqdm(enumerate(list(output.keys()))):
            if image_path == "time":
                continue

            str_im_path = str(image_path)
            
            results = list(output[image_path].values())

            total_boxes = [len(v["boxes"]) for v in results]

            gt_boxes = np.array(results[0]["boxes"])
            pred_boxes = np.array(results[1]["boxes"])

            iou_thres = 0.4

            more_boxes_from_pred = get_diff_boxes(gt_boxes, pred_boxes, iou_thres)
            more_boxes_from_gt = get_diff_boxes(pred_boxes, gt_boxes, iou_thres)

            # breakpoint()
            if (len(more_boxes_from_gt) > 0) or (len(more_boxes_from_pred) > 0):
                imgs = []
                for res in results:
                    imgs.append(draw(str_im_path, res))
                
                im_out = cv2.hconcat(imgs)

                scale_percent = 50 # percent of original size
                width = int(im_out.shape[1] * scale_percent / 100)
                height = int(im_out.shape[0] * scale_percent / 100)
                dim = (width, height)
                im_out = cv2.resize(im_out, dim)

                im_diff = cv2.imread(str_im_path)
                for box in more_boxes_from_gt:
                    x1, y1, x2, y2 = box
                    im_diff = cv2.circle(im_diff, (int((x1+x2)/2), int((y1+y2)/2)), int(min([x2-x1, y2-y1])/2) - 4, (0, 0, 255), thickness=2)

                for box in more_boxes_from_pred:
                    x1, y1, x2, y2 = box
                    im_diff = cv2.circle(im_diff, (int((x1+x2)/2), int((y1+y2)/2)), int(min([x2-x1, y2-y1])/2) - 4, (0, 255, 0), thickness=2)

                im_final = cv2.vconcat([im_out, im_diff])

                path = os.path.join(output_diff_dir, str(index) + ".jpg")
                cv2.imwrite(path, im_final)

                meta_file.write("{} {} \n".format(str(image_path), str(index) + ".jpg"))

                count_diff += 1

        print("Number of difference: ", count_diff)
        meta_file.close()