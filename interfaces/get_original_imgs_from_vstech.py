import shutil
import os
from pathlib import Path


prefix = "{}_{}"
image_dir = "D:\Projects\VSTech\yolov5\outputs\drive-download-20220805T014341Z-001"
folder_name = image_dir.split("\\")[-1]
output_dir = "D:\Projects\VSTech\yolov5\outputs\data_check_august"

jpg_images = list(Path(image_dir).glob("**\*.jpg"))

print("Total images: ", len(jpg_images)/3)

for path in jpg_images:
    if "_fill" in path.stem or "overlay" in path.stem:
        continue
    main_folder_index = list(path.parts).index(folder_name)
    date_folder = path.parts[main_folder_index + 1]
    date_suffix = date_folder.replace("-","")

    image_stem = path.stem
    classify_folder = path.parts[main_folder_index + 3]
    
    if classify_folder == "đúng": 
        classify_index = 1 
    elif classify_folder == "sai":
        classify_index = 0 
    else:
        continue
    
    prefix_name =  prefix.format(date_suffix, classify_index)
    shutil.copyfile(str(path), os.path.join(output_dir, prefix_name + "_" + image_stem + ".jpg"))
