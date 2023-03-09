import glob
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from explore import load_dicom, MODALITIES


class BrainImageDataSet(Dataset):
    """ Brain image dataset. """

    # initialize train and test dataset, with randomness or not.
    def __init__(self, label_csv_path, train_image_dir, modality, mode, split=None, train_test_split=0.8,
                 transform=None):
        self.label_csv_path = label_csv_path
        self.train_image_dir = train_image_dir
        self.split = split
        self.modality = modality
        self.transform = transform
        self.mode = mode
        train_df = pd.read_csv(self.label_csv_path)
        for index in train_df.index:
            train_df.iloc[index, 0] = str(train_df.iloc[index, 0]).zfill(5)
        self.train_df = train_df

        if self.split == 'train':
            self.ids = [a.split("/")[-1] for a in sorted(glob.glob(train_image_dir + "/*"))]
            self.ids = self.ids[:int(len(self.ids) * train_test_split)]  # first 20% as validation
            print(self.ids)

        elif self.split == 'test':
            self.ids = [a.split("/")[-1] for a in sorted(glob.glob(train_image_dir + "/*"))]
            self.ids = self.ids[int(len(self.ids) * train_test_split):]  # last 80% as validation
        else:
            self.ids = [a.split("/")[-1] for a in sorted(glob.glob(train_image_dir + "/*"))]

        if self.mode == 1:
            img_list = list()
            index_mapper = {}
            index_count = 0
            for i in range(len(train_df)):
                subject_dir_path = os.path.join(self.train_image_dir, str(train_df.iloc[i]['BraTS21ID']).zfill(5))
                img_dir_path = os.path.join(subject_dir_path, self.modality)
                t_paths = sorted(
                    glob.glob(os.path.join(img_dir_path, "*")),
                    key=lambda x: int(x[:-4].split("-")[-1]),
                )
                # choose the img in the middle 20 percent
                percentage = 0.1
                img_start_index = int(len(t_paths) * (0.5 - percentage))
                img_end_index = int(len(t_paths) * (0.5 + percentage))
                for img_index in range(img_start_index, img_end_index):
                    img_path = t_paths[img_index]
                    cur_image = load_dicom(img_path)
                    img_list.append(cur_image)

                    index_mapper[index_count] = i
                    index_count = index_count + 1
            self.img_list = img_list
            self.index_mapper = index_mapper

    def __len__(self):
        train_df = pd.read_csv(self.label_csv_path)

        if self.mode == 0:
            return len(self.ids)
        if self.mode == 1:
            img_num = 0
            for i in range(len(train_df)):
                subject_dir_path = os.path.join(self.train_image_dir, str(train_df.iloc[i]['BraTS21ID']).zfill(5))
                img_dir_path = os.path.join(subject_dir_path, self.modality)
                t_paths = sorted(
                    glob.glob(os.path.join(img_dir_path, "*")),
                    key=lambda x: int(x[:-4].split("-")[-1]),
                )
                # choose the img in the middle 20 percent
                percentage = 0.1
                img_start_index = int(len(t_paths) * (0.5 - percentage))
                img_end_index = int(len(t_paths) * (0.5 + percentage))
                img_num = img_num + img_end_index - img_start_index + 1
            return img_num

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        if self.mode == 0:
            # should change index
            cur_index = self.ids[index]
            subject_dir_path = os.path.join(self.train_image_dir, cur_index)
            img_dir_path = os.path.join(subject_dir_path, self.modality)
            t_paths = sorted(
                glob.glob(os.path.join(img_dir_path, "*")),
                key=lambda x: int(x[:-4].split("-")[-1]),
            )
            # choose the img in the middle
            img_path = t_paths[int(len(t_paths) * 0.5)]
            image = load_dicom(img_path)
            # label here stands for MGMT value
            # print(self.train_df['BraTS21ID'] == cur_index)
            row = self.train_df.loc[self.train_df['BraTS21ID'] == cur_index]
            label = row['MGMT_value'].values.item()
            if self.transform:
                image = self.transform(image)
            return image, label
        if self.mode == 1:

            image = self.img_list[index]
            # TODO: wrong index, need to be changed.
            label = self.train_df.iloc[self.index_mapper[index]]['MGMT_value']
            sample = {'image': image, 'label': label}
            if self.transform:
                sample = self.transform(sample)
            return sample


def load_data_path():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, 'data')
    label_csv_path = os.path.join(data_dir, 'train_labels.csv')
    train_image_dir = os.path.join(data_dir, 'train')
    return label_csv_path, train_image_dir


def batch_mean_and_sd(loader):
    """Calculate mean and standard of given dataset.

    It is assumed that different batches have different statistical distribution.
    Therefore, we calculate mean and std for each batch first,
    and then we could generate mean and std for the whole dataset.

    :param loader: Dataloader.
    :return: Mean and standard deviation for the dataset.
    """
    cnt = 0
    fst_moment = torch.empty(1)
    snd_moment = torch.empty(1)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(
        snd_moment - fst_moment ** 2)
    return mean, std


def make_dataset():
    # TODO: Calculate mean and std for our costumed dataset
    pre_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))]
    )
    label_csv_path, train_image_dir = load_data_path()

    brain_image_dataset = BrainImageDataSet(label_csv_path, train_image_dir, MODALITIES[0], 0, transform=pre_transform)

    dataloader = DataLoader(brain_image_dataset, batch_size=64, shuffle=True)

    mean, std = batch_mean_and_sd(dataloader)
    print("mean and std: \n", mean, std)

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            # transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # TODO: need to consider different modalities, different methods. Here we just define two methods. One is select
    #  the middle image from one modality of a subject, the other is selected the middle 20 percent images.
    train_set = BrainImageDataSet(label_csv_path, train_image_dir, MODALITIES[0], 0, split='train',
                                  transform=data_transforms['train'])
    print(len(train_set)
          )
    test_set = BrainImageDataSet(label_csv_path, train_image_dir,
                                 MODALITIES[0], 0, split='test', transform=data_transforms['test'])
    print(len(test_set))
    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)

    # TODO: data augumentation and train test split. K-fold Cross validation
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")
    print("....")
    return train_set, test_set


if __name__ == '__main__':
    make_dataset()
