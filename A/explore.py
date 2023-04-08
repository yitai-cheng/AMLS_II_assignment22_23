import math
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
    # count the number of images for all modalities of each subject
    img_num_list = list(list())

    brats21id_list = train_df['BraTS21ID'].tolist()
    subject_list = list()
    for brats21id in brats21id_list:
        subject_list.append(str(brats21id).zfill(5))

    for subject_name in subject_list:
        subject_dir = os.path.join(train_image_dir, subject_name)
        subject_img_num_list = list()
        for modality in MODALITIES:
            cur_mod_img_num = len(os.listdir(os.path.join(subject_dir, modality)))
            subject_img_num_list.append(cur_mod_img_num)
        img_num_list.append(subject_img_num_list)

    img_num_df = pd.DataFrame(img_num_list, columns=['FLAIR_count', 'T1w_count', 'T1wCE_count', 'T2w_count'])
    img_num_df['Subject_ID'] = subject_list
    img_num_df['MGMT_Value'] = train_df['MGMT_value']
    print(img_num_df)
    # show statistics
    print(img_num_df.describe())

    # plot images for all 4 modalities of a subject. The image for a modality is chosen from the middle of the image
    # group.
    for i in random.sample(range(train_df.shape[0]), 10):
        _brats21id = train_df.iloc[i]["BraTS21ID"]
        _mgmt_value = train_df.iloc[i]["MGMT_value"]
        visualize_sample_by_category(brats21id=_brats21id, slice_i=0.5, mgmt_value=_mgmt_value,
                                     img_path=train_image_dir,
                                     types=MODALITIES)

    # plot the image sequence of a given subject
    sample_id = 582
    subject_id = img_num_df.iloc[sample_id]['Subject_ID']
    visualize_sample_by_sequence(img_path=train_image_dir, subject_id=subject_id)

    # visualize distribution of number of DCM file for each modality
    fig = plt.figure(figsize=(25, 40))
    for i, modality in enumerate(MODALITIES):
        ax = plt.subplot(4, 1, i + 1)
        plt.xticks(rotation=70)
        sns.countplot(x=img_num_df[modality + '_count'], ax=ax)
        ax.set_title("Distribution of number of DCM file in {} modalitys".format(modality),
                     fontsize=18, color="#0b0a2d")

    demo_dir = os.path.join(os.path.dirname(data_dir), 'demo')
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)
    plt.savefig(os.path.join(demo_dir, 'number_distribution'))


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


def visualize_sample_by_category(
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


def visualize_sample_by_sequence(img_path, subject_id):
    patient_path = os.path.join(
        img_path,
        subject_id,
    )

    paths = sorted(
        glob.glob(os.path.join(patient_path, MODALITIES[0], "*")),
        key=lambda x: int(x[:-4].split("-")[-1]),
    )

    img_num = len(paths)

    col_num = 10
    row_num = math.ceil(img_num / col_num)
    path_index = 1

    plt.figure(figsize=(16, 12))

    for i in range(img_num):
        img = load_dicom(paths[i])
        plt.subplot(row_num, col_num, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        path_index = path_index + 1

    demo_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'demo')
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)
    plt.subplots_adjust(wspace=0.01)
    plt.savefig(os.path.join(demo_dir, 'sequence'))


if __name__ == '__main__':
    explore_dataset()
