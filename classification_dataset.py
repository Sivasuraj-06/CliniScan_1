import os
import shutil
import random
import pandas as pd
from sklearn.model_selection import train_test_split

png_dir = "png_train"  
csv_path = "vinbigdata-chest-xray-abnormalities-detection/train.csv"
output_dir = "cls_dataset_binary"

VAL_RATIO = 0.2
random.seed(42)

df = pd.read_csv(csv_path)

df["binary_label"] = df["class_name"].apply(lambda x: "normal" if x == "No finding" else "abnormal")

normal_ids = df[df["binary_label"] == "normal"]["image_id"].unique().tolist()
abnormal_ids = df[df["binary_label"] == "abnormal"]["image_id"].unique().tolist()

if len(normal_ids) > len(abnormal_ids):
    normal_ids = random.sample(normal_ids, len(abnormal_ids))

all_ids = normal_ids + abnormal_ids
labels_map = {img_id: ("abnormal" if img_id in abnormal_ids else "normal") for img_id in all_ids}

stratify_labels = [1 if labels_map[i] == "abnormal" else 0 for i in all_ids]
train_ids, val_ids = train_test_split(all_ids, test_size=VAL_RATIO, random_state=42, stratify=stratify_labels)

def copy_files(ids, split):
    for img_id in ids:
        img_src = os.path.join(png_dir, f"{img_id}.png")
        if not os.path.exists(img_src):
            continue
        label = labels_map[img_id]
        dst_dir = os.path.join(output_dir, split, label)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(img_src, os.path.join(dst_dir, f"{img_id}.png"))


copy_files(train_ids, "train")
copy_files(val_ids, "val")

print(f"✅ Balanced binary classification dataset prepared:")
print(f"   Train: {len(train_ids)} images ({len([i for i in train_ids if labels_map[i]=='normal'])} normal, {len([i for i in train_ids if labels_map[i]=='abnormal'])} abnormal)")
print(f"   Val:   {len(val_ids)} images ({len([i for i in val_ids if labels_map[i]=='normal'])} normal, {len([i for i in val_ids if labels_map[i]=='abnormal'])} abnormal)")
