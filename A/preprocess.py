"""This file contains functions of data preprocessing.

We define PyTorch dataset class.
We select data by different data selection modes.
Images are loaded and transformed by flipping, crop and resizing.
The whole dataset is split into train, validation and test set.

"""
import glob
import os
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from A.constants import DATA_TRANSFORMS
from A.explore import load_dicom, MODALITIES


class BrainImageDataSet(Dataset):
    """This is the pytorch dataset Class.

    We construct the dataset in the form of pytorch dataset.
    The dataset is split into train, validation and test set.
    Several modes of constructing dataset could be used.
    mode 0: choose one image located in the middle of a given modality.
    mode 1: choose images in the middle 20 percent of a given modality.
    # mode 2: choose 64 images whose average pixel intensity is greatest among all modalities.
    mode 2: choose one image located in the middle of each modality and combine them into an array
    mode 3: choose 20 images located in the middle of each modality and combine them into an array


    """

    # initialize train and test dataset, with randomness or not.
    def __init__(self, label_csv_path, train_image_dir, identities, mode, modality=None,
                 transform=None):
        self.label_csv_path = label_csv_path
        self.train_image_dir = train_image_dir
        self.modality = modality
        self.transform = transform
        self.mode = mode
        train_df = pd.read_csv(self.label_csv_path)
        for index in train_df.index:
            train_df.iloc[index, 0] = str(train_df.iloc[index, 0]).zfill(5)
        self.train_df = train_df
        self.label_list = list()
        self.identities = identities
        # print(self.identities)

        if self.mode == 0:
            img_path_list = list()
            for i in range(len(self.identities)):
                identity = self.identities[i]
                subject_dir_path = os.path.join(self.train_image_dir, identity)
                img_dir_path = os.path.join(subject_dir_path, self.modality)
                t_paths = sorted(
                    glob.glob(os.path.join(img_dir_path, "*")),
                    key=lambda x: int(x[:-4].split("-")[-1]),
                )
                # choose the img in the middle
                img_path = t_paths[int(len(t_paths) * 0.5)]
                img_path_list.append(img_path)
                self.label_list.append(train_df.iloc[i]['MGMT_value'])
            self.img_path_list = img_path_list

        if self.mode == 1:
            img_path_list = list()
            index_mapper = {}
            index_count = 0
            img_num = 0
            for i in range(len(self.identities)):
                identity = self.identities[i]
                subject_dir_path = os.path.join(self.train_image_dir, identity)
                img_dir_path = os.path.join(subject_dir_path, self.modality)
                t_paths = sorted(
                    glob.glob(os.path.join(img_dir_path, "*")),
                    key=lambda x: int(x[:-4].split("-")[-1]),
                )
                # TODO: choose the img in the middle 20 percent / 20 items
                percentage = 0.1
                img_mid_index = int(len(t_paths) * 0.5)
                img_start_index = img_mid_index - 5
                img_end_index = img_start_index + 10
                for img_index in range(img_start_index, img_end_index):
                    img_path = t_paths[img_index]
                    # cur_image = load_dicom(img_path)
                    img_path_list.append(img_path)
                    index_mapper[index_count] = i
                    index_count = index_count + 1
                    self.label_list.append(train_df.iloc[i]['MGMT_value'])
                img_num = img_num + img_end_index - img_start_index
            self.img_path_list = img_path_list
            self.index_mapper = index_mapper
            self.img_num = img_num

        if self.mode == 2:
            img_path_list = list()
            for i in range(len(self.identities)):
                identity = self.identities[i]
                subject_dir_path = os.path.join(self.train_image_dir, identity)
                cur_img_list = list()
                for modality in MODALITIES:
                    img_dir_path = os.path.join(subject_dir_path, modality)
                    t_paths = sorted(
                        glob.glob(os.path.join(img_dir_path, "*")),
                        key=lambda x: int(x[:-4].split("-")[-1]),
                    )
                    # choose the img in the middle
                    img_path = t_paths[int(len(t_paths) * 0.5)]
                    cur_img_list.append(img_path)
                img_path_list.append(cur_img_list)
                self.label_list.append(train_df.iloc[i]['MGMT_value'])
            self.img_path_list = img_path_list

        if self.mode == 3:
            img_path_list = list()
            index_mapper = {}
            index_count = 0
            img_num = 0
            for i in range(len(self.identities)):
                identity = self.identities[i]
                subject_dir_path = os.path.join(self.train_image_dir, identity)
                cur_img_list = list()

                for j in range(len(MODALITIES)):
                    modality = MODALITIES[j]
                    img_dir_path = os.path.join(subject_dir_path, modality)
                    t_paths = sorted(
                        glob.glob(os.path.join(img_dir_path, "*")),
                        key=lambda x: int(x[:-4].split("-")[-1]),
                    )
                    # choose the img in the middle
                    # TODO: choose the img in the middle 20 percent / 20 items
                    img_mid_index = int(len(t_paths) * 0.5)
                    img_start_index = img_mid_index - 5
                    img_end_index = img_start_index + 10
                    cur_img_modal_list = list()
                    for img_index in range(img_start_index, img_end_index):
                        img_path = t_paths[img_index]
                        # cur_image = load_dicom(img_path)
                        cur_img_modal_list.append(img_path)
                        if j == 0:
                            index_mapper[index_count] = i
                            index_count = index_count + 1
                            self.label_list.append(train_df.iloc[i]['MGMT_value'])
                    if j == 0:
                        img_num = img_num + img_end_index - img_start_index
                    cur_img_list.append(cur_img_modal_list)
                img_path_list.append(cur_img_list)
            self.img_path_list = img_path_list
            self.index_mapper = index_mapper
            self.img_num = img_num

    def __len__(self):
        train_df = pd.read_csv(self.label_csv_path)

        if self.mode == 0:
            return len(self.identities)
        if self.mode == 1:
            return self.img_num
        if self.mode == 2:
            return len(self.identities)
        if self.mode == 3:
            return self.img_num

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        image = None
        label = None
        if self.mode == 0:

            image = load_dicom(self.img_path_list[index])
            # label here stands for MGMT value
            # print(self.train_df['BraTS21ID'] == cur_index)
            # row = self.train_df.loc[self.train_df['BraTS21ID'] == cur_index]

            label = self.train_df.iloc[index]['MGMT_value']
            image = self.transform(image)

        elif self.mode == 1:
            image = load_dicom(self.img_path_list[index])
            # TODO: wrong index, need to be changed.
            label = self.train_df.iloc[self.index_mapper[index]]['MGMT_value']
            image = self.transform(image)

        elif self.mode == 2:
            # build the final image.
            modality_img_path_list = self.img_path_list[index]
            all_modality_img_list = list()
            for single_modality_img_path in modality_img_path_list:
                single_modality_img = load_dicom(single_modality_img_path)
                transformed_single_modality_img = self.transform(single_modality_img)
                all_modality_img_list.append(torch.squeeze(transformed_single_modality_img))

            image = torch.stack(all_modality_img_list, 0)

            label = self.train_df.iloc[index]['MGMT_value']

        elif self.mode == 3:
            # TODO: 10 is hard coded, need to change
            subject_index = index // 10
            img_frame_index = index % 10
            # if subject_index >= len(self.img_path_list):
            #     print(subject_index)
            modality_img_path_list = self.img_path_list[subject_index]
            all_modality_img_list = list()

            for cur_modality_img_path_list in modality_img_path_list:
                single_modality_img_path = cur_modality_img_path_list[img_frame_index]
                single_modality_img = load_dicom(single_modality_img_path)
                transformed_single_modality_img = self.transform(single_modality_img)
                all_modality_img_list.append(torch.squeeze(transformed_single_modality_img))
            image = torch.stack(all_modality_img_list, 0)

            label = self.train_df.iloc[self.index_mapper[index]]['MGMT_value']

        return image, label


def load_data_path():
    """Load data path.

    :return: path for image directory and label csv file.
    """
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


def data_preprocess(data_make_mode, modality=None):
    """

    :param data_make_mode: define the way of data selection
    :param modality: If data_make_mode is 0 or 1, then modality is needed because we only need to select one modality.
    :return: train, validation and test set.
    """
    label_csv_path, train_image_dir = load_data_path()

    # TODO: Calculate mean and std for our costumed dataset
    # pre_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((256, 256))]
    # )
    # # define different criterion for making our customed dataset
    # brain_image_dataset = BrainImageDataSet(label_csv_path, train_image_dir, data_make_mode, modality=modality,
    #                                         transform=pre_transform)
    #
    # dataloader = DataLoader(brain_image_dataset, batch_size=64, shuffle=True)

    # mean, std = batch_mean_and_sd(dataloader)
    # print("mean and std: \n", mean, std)

    # TODO: need to consider different modalities, different methods. Here we just define two methods. One is select
    #  the middle image from one modality of a subject, the other is selected the middle 20 percent images.
    TEST_SIZE = 0.2
    SEED = 42

    train_df = pd.read_csv(label_csv_path)
    for index in train_df.index:
        train_df.iloc[index, 0] = str(train_df.iloc[index, 0]).zfill(5)

    # split train and test set in a stratified fashion
    train_val_indices, test_indices, _, _ = train_test_split(
        range(len(train_df)),
        train_df['MGMT_value'],
        stratify=train_df['MGMT_value'],
        test_size=TEST_SIZE,
        random_state=SEED
    )
    test_ids = train_df.loc[test_indices, 'BraTS21ID'].tolist()

    # build k-fold cross validation set
    k_folder = StratifiedKFold(n_splits=4, random_state=13, shuffle=True)
    X = train_df.loc[train_val_indices, 'BraTS21ID']
    y = train_df.loc[train_val_indices, 'MGMT_value']
    fold_no = 1
    k_fold_cv_ids_list = list()
    for train_indices, val_indices in k_folder.split(X, y):
        train_ids = train_df.loc[train_indices, 'BraTS21ID'].tolist()
        val_ids = train_df.loc[val_indices, 'BraTS21ID'].tolist()
        k_fold_cv_ids_list.append({'train_ids': train_ids, 'val_ids': val_ids})

    data_transforms = DATA_TRANSFORMS

    k_fold_cv_list = list()

    for cur_fold_cv_ids in k_fold_cv_ids_list:
        train_ids = cur_fold_cv_ids['train_ids']
        val_ids = cur_fold_cv_ids['val_ids']
        training_set = BrainImageDataSet(label_csv_path, train_image_dir, train_ids, data_make_mode,
                                         modality=modality,
                                         transform=data_transforms['train'])
        # print(len(training_set))

        validation_set = BrainImageDataSet(label_csv_path, train_image_dir, val_ids, data_make_mode,
                                           modality=modality,
                                           transform=data_transforms['val'])
        k_fold_cv_list.append({'training_set': training_set, 'validation_set': validation_set})

    test_set = BrainImageDataSet(label_csv_path, train_image_dir, test_ids, data_make_mode,
                                 modality=modality,
                                 transform=data_transforms['test'])
    # print(len(test_set))


    # generate indices: instead of the actual data we pass in integers instead, split train into train and val set
    # train_indices, val_indices, _, _ = train_test_split(
    #     range(len(train_set)),
    #     train_set.label_list,
    #     stratify=train_set.label_list,
    #     test_size=TEST_SIZE,
    #     random_state=SEED
    # )

    # # generate subset based on indices
    # training_set = Subset(train_set, train_indices)
    # validation_set = Subset(train_set, val_indices)

    #  ------------ to be deleted ----------------
    # train_dataloader = DataLoader(test_set, shuffle=True)
    # train_features, train_labels = next(iter(train_dataloader))

    #
    # test_dataloader = None
    # # # TODO: data augmentation and train test split. K-fold Cross validation
    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # img = train_features[0].squeeze()
    # label = train_labels[0]
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # print(f"Label: {label}")
    # print("....")
    # #  ------------ to be deleted ----------------

    return k_fold_cv_list, test_set


# if __name__ == '__main__':


