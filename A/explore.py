import os
import json
import glob
import random
import collections
import numpy as np
import pandas as pd
import pydicom
# import cv2
import matplotlib.pyplot as plt
import seaborn as sns

MODALITIES = ['FLAIR', 'T1w', 'T1wCE', 'T2w']


def explore_dataset():

    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, 'data')
    label_csv_ath = os.path.join(data_dir, 'train_labels.csv')
    train_df = pd.read_csv(label_csv_ath)

    train_image_dir = os.path.join(data_dir, 'train')
    print(train_image_dir)
    print(len(os.listdir(train_image_dir)))

    # count the number of images for all modalities of each subject
    img_num_list = list(list())
    print(img_num_list)

    subject_list = os.listdir(train_image_dir)
    if '.DS_Store' in subject_list:
        subject_list.remove('.DS_Store')

    for subject_name in subject_list:
        subject_dir = os.path.join(train_image_dir, subject_name)
        subject_img_num_list = list()
        for modality in MODALITIES:
            cur_mod_img_num = len(os.listdir(os.path.join(subject_dir, modality)))
            subject_img_num_list.append(cur_mod_img_num)
        img_num_list.append(subject_img_num_list)

    img_num_df = pd.DataFrame(img_num_list, columns=MODALITIES)
    img_num_df['Subject'] = subject_list
    print(img_num_df)
    # show statistics
    print(img_num_df.describe())

    for i in random.sample(range(train_df.shape[0]), 10):
        _brats21id = train_df.iloc[i]["BraTS21ID"]
        _mgmt_value = train_df.iloc[i]["MGMT_value"]
        visualize_sample(brats21id=_brats21id, slice_i=0.5, mgmt_value=_mgmt_value, img_path=train_image_dir, types=MODALITIES)


def load_dicom(path):
    """Load image data.

    Load dicom image file by pydicom library function.

    :param path: image file path.
    :return: transformed image data in the form of int matrix.

    """
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


def visualize_sample(
        brats21id,
        slice_i,
        mgmt_value,
        img_path,
        types
):
    plt.figure(figsize=(16, 5))
    patient_path = os.path.join(
        img_path,
        str(brats21id).zfill(5),
    )
    for i, t in enumerate(types, 1):
        t_paths = sorted(
            glob.glob(os.path.join(patient_path, t, "*")),
            key=lambda x: int(x[:-4].split("-")[-1]),
        )
        data = load_dicom(t_paths[int(len(t_paths) * slice_i)])
        plt.subplot(1, 4, i)
        plt.imshow(data, cmap="gray")
        plt.title(f"{t}", fontsize=16)
        plt.axis("off")

    plt.suptitle(f"MGMT_value: {mgmt_value}", fontsize=16)
    img_name = str(brats21id).zfill(5)

    demo_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'demo')
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)

    plt.savefig(os.path.join(demo_dir, img_name))



if __name__ == '__main__':

    explore_dataset()
