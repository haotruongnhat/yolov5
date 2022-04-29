from pathlib import Path


label_path_files = Path("D:\Projects\VSTech\yolov5\outputs\data_27_04_1c_with_bg\labels\\train").glob("*.txt")
for label_file in label_path_files:
    with open(str(label_file), 'rb') as fp:
        for count, line in enumerate(fp):
                pass
        count_all = count + 1
        if count_all in [115, 150, 95, 36, 68, 60, 44, 95, 76]:
            continue

        print("File", label_file.stem, "Total Lines", count_all)