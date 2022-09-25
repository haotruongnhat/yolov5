import shutil
from pathlib import Path
import os

path = "D:\Projects\VSTech\yolov5\outputs\data_test_on_May\data_full"
images = []

output_dir = "D:\Projects\VSTech\yolov5\outputs\data_test_on_May\data_full_recheck"
for index, p in enumerate(list(Path(path).glob("**/*.jpg"))):
    if ("overlay" in str(p)) or ("fill" in str(p)):
        continue

    date = p.parents[1].name.replace("-", "_")
    shutil.copyfile(str(p), os.path.join(output_dir, date + "_" + p.name))
