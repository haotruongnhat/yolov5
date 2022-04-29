import pandas as pd 
import os 
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil

DELIMITER = "\\"

def split_img_label(data_train,data_test, output_parent_folder):
    train_image_folder = os.path.join(output_parent_folder, "images", "train")
    test_image_folder = os.path.join(output_parent_folder, "images", "test")
    train_label_folder = os.path.join(output_parent_folder, "labels", "train")
    test_label_folder = os.path.join(output_parent_folder, "labels", "test")

    os.makedirs(train_image_folder, exist_ok = True)
    os.makedirs(test_image_folder, exist_ok = True)
    os.makedirs(train_label_folder,exist_ok =  True)
    os.makedirs(test_label_folder, exist_ok = True)
    
    train_ind=list(data_train.index)
    test_ind=list(data_test.index)
    
    # Train folder
    for i in tqdm(range(len(train_ind))):
        shutil.copyfile(data_train[train_ind[i]], os.path.join(train_image_folder, data_train[train_ind[i]].split(DELIMITER)[-1]))
        if "bg" in data_train[train_ind[i]].split(DELIMITER)[-1]:
            continue
        shutil.copyfile(data_train[train_ind[i]].split('.jpg')[0]+'.txt', os.path.join(train_label_folder, data_train[train_ind[i]].split(DELIMITER)[-1].split(".jpg")[0]+'.txt'))

    # Test folder
    for i in tqdm(range(len(test_ind))):
        shutil.copyfile(data_test[test_ind[i]], os.path.join(test_image_folder, data_test[test_ind[i]].split(DELIMITER)[-1]))
        if "bg" in data_test[test_ind[i]].split(DELIMITER)[-1]:
            continue
        shutil.copyfile(data_test[test_ind[i]].split('.jpg')[0]+'.txt', os.path.join(test_label_folder, data_test[test_ind[i]].split(DELIMITER)[-1].split(".jpg")[0] +'.txt'))

PATH = 'D:\Projects\VSTech\yolov5\outputs\data_full\data'
list_img=[img for img in os.listdir(PATH) if img.endswith('.jpg')==True]
list_txt=[img for img in os.listdir(PATH) if img.endswith('.txt')==True]

path_img=[]

for i in range (len(list_img)):
    path_img.append(os.path.join(PATH, list_img[i]))
    
df = pd.DataFrame(path_img)

# split 
data_train, data_test, labels_train, labels_test = train_test_split(df[0], df.index, test_size=0.20, random_state=42)

output_parent_folder = "D:\Projects\VSTech\yolov5\outputs\data_full\output"
# Function split 
split_img_label(data_train, data_test, output_parent_folder)

shutil.copyfile(os.path.join(PATH, "classes.txt"), os.path.join(output_parent_folder, "labels", "test", "classes.txt"))
shutil.copyfile(os.path.join(PATH, "classes.txt"), os.path.join(output_parent_folder, "labels", "train", "classes.txt"))

