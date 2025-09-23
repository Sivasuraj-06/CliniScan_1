import os
import pydicom
import cv2
from tqdm import tqdm

def dicom_to_png(dicom_path, output_path):
    dicom = pydicom.dcmread(dicom_path)
    img = dicom.pixel_array
    img = cv2.convertScaleAbs(img, alpha=(255.0/img.max()))  
    cv2.imwrite(output_path, img)

def batch_convert(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in tqdm(os.listdir(input_dir)):
        if file.endswith(".dicom") or file.endswith(".dcm"):
            dicom_path = os.path.join(input_dir, file)
            png_path = os.path.join(output_dir, file.replace(".dicom", ".png").replace(".dcm", ".png"))
            dicom_to_png(dicom_path, png_path)

if __name__ == "__main__":
    batch_convert("vinbigdata-chest-xray-abnormalities-detection/train", "png_train")
    batch_convert("vinbigdata-chest-xray-abnormalities-detection/test", "png_test")
