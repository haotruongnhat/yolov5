#!/usr/bin/python
import socket
import pickle
import json 

def chunk_send(data, socket, addr):
    buf = 4096
    i = 0
    j = buf
    msg = pickle.dumps(data)
    packet_len = pickle.dumps(len(msg))
    socket.sendto(packet_len, addr)

    # Send pickle
    while(i<len(msg)):
        if j>(len(msg)-1):
            j=(len(msg))
            
        ###send pickle chunks
        socket.sendto(msg[i:j], addr)

        i += buf
        j += buf
        
def chunk_recv(socket):
    buf = 4096
    Dlen = int(pickle.loads(socket.recv(buf)))
    pkl_str = bytearray()
    for i in range(0,int(int(Dlen)/buf)+1):
        data = socket.recv(buf)
        pkl_str += data

    data = pickle.loads(pkl_str)
    
    return data

def exchange(client_port, server_port, im_path, conf_thres=0.6, nms=0.25, eps_m=2, eps_o= 0, eps_samples=3):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", int(client_port)))

        detect_config = dict(conf_thres=conf_thres, nms=nms)
        filter_config = dict(eps_enable=True, eps_m=eps_m, eps_o=eps_o, num_samples=eps_samples)
        config = dict(im_path = im_path, detect_config=detect_config, filter_config=filter_config)

        chunk_send(config, s, ("127.0.0.1", int(server_port)))

        result_dict = chunk_recv(s)
        ## Post-process
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
    from pathlib import Path
    import time
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import cv2
    from sklearn.cluster import DBSCAN
    import torch
    from tqdm import tqdm


    # source_im_path = "D:\Projects\VSTech\yolov5\outputs\\test_images"
    # source_im_path = "D:\Projects\VSTech\yolov5\outputs\output_2704_20epoch_freeze_23\\final_fail"
    
    source_im_path = "D:\Projects\VSTech\yolov5\outputs\output_2704_20epoch_freeze_23\\all_retrained_2\\errors"

    images = list(Path(source_im_path).glob("**\*.jpg"))

    ip = "127.0.0.1"
    server_addr = 50000
    client_addr = 50004

    def draw(image, result):
        im = cv2.imread(image)
        for box, cls_index in zip(result["boxes"], result["classes"]):
            score_str = list(box.keys())[0][:4]
            x1, y1, x2, y2 = list(box.values())[0]

            # im2 = im.copy()

            # im = cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, cls_index*255), 2)
            cv2.putText(im, score_str, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2)

            im = cv2.circle(im, (int((x1+x2)/2), int((y1+y2)/2)), int(min([x2-x1, y2-y1])/2) - 4, (255, 0, 0), thickness=-1)
            # im = cv2.vconcat((im, im2))

        cv2.putText(im, str(len(result["boxes"])), (50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), thickness=3)

        return im

    def xyxy2xywh(x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    def save_txt(path, result, image_shape):
        gn = np.array(image_shape)[[1, 0, 1, 0]]
        for box, cls in zip(result["boxes"], result["classes"]):
            xyxy = np.array(list(box.values())[0])
            xywh = (xyxy2xywh(xyxy.reshape(1, -1)) / gn).reshape(-1) .tolist()  # normalized xywh
            line = cls, *xywh # label format
            with open(path, 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

    def save_classes_txt(path):
        with open(path, 'a') as f:
            f.write("thep" + '\n')


    output_path = "D:\Projects\VSTech\yolov5\outputs\output_2704_20epoch_freeze_23\\all_retrained_2\\errors_retrained"
    output_im_path = os.path.join(output_path, "result_im")
    output_label_path = os.path.join(output_path, "labels")
    failed_image_path = os.path.join(output_path, "fail")

    # error_conf_0_4_images = ["01042022_2_2.jpg", "01042022_2_25.jpg", "01042022_2_117.jpg", "01042022_2_120.jpg",
    #                 "01042022_2_143.jpg", "01042022_2_149.jpg", "01042022_2_158.jpg", "01042022_2_162.jpg"
    #                 "01042022_2_164.jpg"]; images = error_images
    try:
        os.makedirs(output_path)
    except: 
        pass

    try:
        os.makedirs(output_im_path)
    except: 
        pass

    try:
        os.makedirs(output_label_path)
    except: 
        pass

    try:
        os.makedirs(failed_image_path)
    except: 
        pass

    # error_conf_0_4_images = ["01042022_2_2.jpg", "02042022_2_Nhi-47.jpg"]; images = error_conf_0_4_images
    # images = [
    #         "01042022_2_113.jpg",
    #         "01042022_2_184.jpg",
    #         "01042022_2_259.jpg",
    #         "01042022_2_305.jpg"]
    failed_count = 0
    print("Total images:", len(list(images)))
    for index, im in tqdm(enumerate(list(images))):
        # if "01042022" not in str(im):
        #     continue

        image_name = Path(im).name
        image_stem = Path(im).stem
        output_file = os.path.join(output_im_path, image_name)
        failed_file = os.path.join(failed_image_path, image_name)

        # source_image = os.path.join(source_im_path, image_name)
        # if os.path.exists(output_file):
        #     print("Skipping", image_name)
        #     continue

        result_dict = exchange(client_addr, server_addr, str(im), conf_thres=0.4, nms=0.5, eps_m=3, eps_o= 0, eps_samples=3)

        data = json.loads(result_dict)
        # if len(data["boxes"]) != 150:
        draw_im = draw(str(im), data)
        if draw_im is None:
            continue
        num_box = len(data["boxes"])
        if num_box < 5:
            continue

        # if num_box in [201, 150, 190, 95, 76, 68, 60, 44, 36]:
        #     continue

        failed_count += 1
        print("Current failed:", failed_count)
        with open(os.path.join(output_path, "miss_150.txt"), 'a') as f:
            f.write(image_name + ' ' + str(num_box) + '\n')
        cv2.imwrite(output_file, draw_im)
        source_im = cv2.imread(str(im))
        cv2.imwrite(failed_file, source_im)
        save_txt(os.path.join(output_label_path, image_stem + ".txt"), data, draw_im.shape)



        # print("Write to:", output_file)

        # addition_image = os.path.join("D:\Projects\VSTech\yolov5\outputs\gpu", image_name)

        # addition_im = cv2.imread(addition_image)
        # stack_im = cv2.vconcat((draw_im, addition_im))


        # bboxes_center = []
        # for box in data["boxes"]:
        #     score_str = list(box.keys())[0][:4]
        #     x1, y1, x2, y2 = list(box.values())[0]

        #     bboxes_center.append([(x1+x2)/2, (y1+y2)/2])

        # if bboxes_center:
        #     bboxes_center = np.array(bboxes_center)
        #     # points = [[504.5, 749.5], [311.0, 485.0], [288.0, 738.5], [55.0, 58.5], [306.5, 258.5], [1051.5, 513.0], [1088.0, 477.0], [664.0, 446.5], [1094.5, 513.0], [953.5, 514.5], [1075.5, 548.0], [383.5, 350.0], [746.0, 530.5], [516.0, 286.0], [1032.0, 550.5], [995.5, 534.5], [891.0, 352.5], [994.5, 412.5], [978.0, 483.0], [900.5, 613.0], [407.5, 464.5], [809.0, 337.5], [732.5, 365.5], [888.0, 574.0], [447.5, 430.0], [924.5, 436.5], [659.5, 410.5], [356.0, 468.0], [699.0, 426.5], [560.0, 469.5], [846.5, 568.5], [1015.0, 496.0], [407.0, 424.5], [1000.0, 577.0], [869.0, 310.0], [608.0, 357.0], [708.0, 519.0], [1034.0, 422.5], [679.5, 548.0], [964.0, 557.5], [735.0, 405.5], [900.0, 467.0], [1062.0, 581.5], [872.0, 388.5], [584.0, 391.0], [359.5, 385.5], [754.0, 260.0], [982.5, 615.0], [506.0, 379.0], [565.5, 427.5], [471.0, 277.5], [941.0, 474.5], [911.5, 397.5], [851.0, 349.5], [601.5, 468.5], [1029.5, 612.0], [1015.0, 378.0], [433.0, 510.0], [1004.0, 451.0], [824.5, 301.5], [884.0, 427.0], [883.0, 534.0], [1045.0, 465.5], [964.0, 440.0], [912.5, 507.5], [447.0, 470.5], [820.0, 375.5], [557.0, 271.0], [487.5, 415.0], [804.0, 565.0], [660.5, 367.5], [638.5, 543.0], [840.0, 412.0], [624.5, 392.0], [1048.0, 354.5], [863.5, 607.5], [543.5, 391.0], [556.5, 520.5], [432.0, 349.5], [831.0, 526.0], [777.5, 482.0], [388.5, 505.0], [622.0, 589.0], [787.0, 530.0], [943.5, 595.5], [914.5, 318.5], [565.5, 354.5], [796.5, 273.0], [956.0, 399.0], [1080.0, 436.0], [779.5, 378.5], [446.0, 390.5], [473.0, 504.0], [583.5, 598.0], [731.0, 449.5], [973.5, 360.0], [523.5, 344.5], [846.0, 267.0], [592.5, 248.0], [560.0, 563.0], [798.5, 410.0], [719.0, 325.5], [854.5, 451.5], [932.5, 360.5], [1062.5, 395.0], [698.5, 387.0], [404.5, 382.0], [680.0, 331.0], [637.0, 329.0], [518.0, 478.0], [739.0, 488.0], [508.0, 519.0], [357.5, 427.0], [762.5, 343.0], [402.5, 589.5], [959.5, 317.0], [588.5, 321.5], [380.0, 545.5], [662.0, 293.0], [924.5, 550.0], [547.0, 311.5], [472.5, 358.5], [449.5, 312.0], [489.5, 318.0], [599.0, 555.0], [616.5, 286.5], [517.5, 560.0], [450.5, 586.5], [763.5, 566.5], [810.5, 447.5], [768.5, 439.0], [407.0, 315.5], [540.0, 597.0], [427.5, 549.5], [697.0, 478.5], [739.5, 603.5], [893.0, 276.5], [760.5, 300.5], [1006.0, 332.5], [696.5, 599.5], [633.0, 247.5], [486.5, 458.5], [676.0, 251.0], [823.0, 485.5], [821.5, 605.0], [470.0, 549.5], [659.0, 583.0], [704.5, 284.0], [779.5, 607.5], [648.0, 502.0], [598.0, 511.0], [490.5, 592.0], [523.5, 430.5], [867.5, 493.5], [722.5, 568.5]]
        #     # bboxes_center = np.array(points)
        #     db = DBSCAN(eps=60, min_samples=3).fit(bboxes_center)
            
        #     filterred_result = {}
        #     filterred_result["boxes"] = []

        #     unq, counts = np.unique(db.labels_, return_counts=True)
        #     largest_blob_index = np.argmax(counts)
        #     largest_index =  unq[largest_blob_index]

        #     for index, value in enumerate(db.labels_):
        #         if value == largest_index:
        #             filterred_result["boxes"].append(data["boxes"][index])

        #     if len(filterred_result["boxes"]) != 150:
        #         draw_im = draw(source_image, filterred_result)

        #         print("Write to:", output_file)

        #         addition_image = os.path.join("D:\Projects\VSTech\yolov5\outputs\gpu", image_name)

        #         addition_im = cv2.imread(addition_image)
        #         stack_im = cv2.vconcat((draw_im, addition_im))

        #         cv2.imwrite(output_file, stack_im)
        # plt.imshow(draw_im)
        # plt.pause(0.001)

        # time.sleep(1)

    save_classes_txt(os.path.join(output_label_path, "classes.txt"))
    print("Sample run: ", index +1, "Diff 150 Count:", failed_count)