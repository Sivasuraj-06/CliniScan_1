import os
import cv2
import pandas as pd
import pydicom

def dicom_to_jpeg(dicom_path, jpeg_path):
    dcm = pydicom.dcmread(dicom_path)
    img = dcm.pixel_array
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    cv2.imwrite(jpeg_path, img)

def convert_dataset(dicom_dir, out_dir, csv_annotations):
    os.makedirs(out_dir, exist_ok=True)
    annots = pd.read_csv(csv_annotations)

    for _, row in annots.iterrows():
        dicom_file = os.path.join(dicom_dir, row['image_id'] + ".dicom")
        jpeg_file = os.path.join(out_dir, row['image_id'] + ".jpeg")

        if os.path.exists(dicom_file):
            dicom_to_jpeg(dicom_file, jpeg_file)
            print(f"Converted: {dicom_file} -> {jpeg_file}")
        else:
            print(f"Warning: {dicom_file} not found, skipping.")

if __name__ == "__main__":
    dicom_folder = "dicom_dir"
    output_folder = "converted_jpegs"
    annotation_csv = "annotations.csv"

    convert_dataset(dicom_folder, output_folder, annotation_csv)
