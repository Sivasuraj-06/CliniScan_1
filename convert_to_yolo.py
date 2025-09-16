import os
import shutil
from PIL import Image


root_dir = "data/chest_xray"           
output_dir = "data/chest_xray_yolo"    
splits = ["train", "val", "test"]     
class_names = ["NORMAL", "PNEUMONIA"]  
class_to_id = {cls: i for i, cls in enumerate(class_names)}


for split in splits:
    split_dir = os.path.join(root_dir, split)
    if not os.path.exists(split_dir):
        continue

    out_images = os.path.join(output_dir, split, "images")
    out_labels = os.path.join(output_dir, split, "labels")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    for cls in os.listdir(split_dir):
        cls_path = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        class_id = class_to_id[cls]

        for img_name in os.listdir(cls_path):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(cls_path, img_name)
            out_img_path = os.path.join(out_images, img_name)
            
            
            shutil.copy(img_path, out_img_path)

            
            with Image.open(img_path) as im:
                w, h = im.size
            x_center, y_center, width, height = 0.5, 0.5, 1.0, 1.0

            label_path = os.path.join(out_labels, img_name.rsplit(".", 1)[0] + ".txt")
            with open(label_path, "w") as f:
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

print("✅ Conversion complete. YOLO-style dataset saved at:", output_dir)


yaml_path = os.path.join(output_dir, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"train: {os.path.join(output_dir, 'train/images')}\n")
    f.write(f"val: {os.path.join(output_dir, 'val/images')}\n")
    f.write(f"test: {os.path.join(output_dir, 'test/images')}\n")
    f.write(f"nc: {len(class_names)}\n")
    f.write(f"names: {class_names}\n")

print("✅ data.yaml created at:", yaml_path)
