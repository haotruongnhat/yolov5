import shutil

image_dir = "D:\Projects\VSTech\yolov5\outputs\output_2704_20epoch_freeze_23\\all_retrained_2\\fail"
label_dir = "D:\Projects\VSTech\yolov5\outputs\output_2704_20epoch_freeze_23\\all_retrained_2\\labels"

output_dir = "D:\Projects\VSTech\yolov5\outputs\output_2704_20epoch_freeze_23\\all_retrained_2\\errors"

with open("D:\Projects\VSTech\yolov5\outputs\output_2704_20epoch_freeze_23\\all_retrained_2\errors.txt") as f:
    image_ids = f.read().strip().split()

import os
from pathlib import Path

jpg_images = list(Path("D:\Projects\VSTech\yolov5\outputs\output_2704_20epoch_freeze_23\\all_retrained_2\\fail").glob("*.jpg"))
for path in jpg_images:
    image_stem = path.stem

    for im_id in image_ids:
        if im_id in image_stem:
            shutil.copyfile(os.path.join(image_dir, image_stem + ".jpg"), os.path.join(output_dir, image_stem + ".jpg"))
            shutil.copyfile(os.path.join(label_dir, image_stem + ".txt"), os.path.join(output_dir, image_stem + ".txt"))