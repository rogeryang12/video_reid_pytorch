import os
import h5py
import random
import numpy as np
from PIL import Image


def train_test(image_root, train_ratio=0.5):
    """
    split the ilids dataset into trainset and testset in dict form
    dataset[pid][cam] = tracklet
    tracklet in list form,  consist of image filename
    :param image_root: the root directory of ilids dataset
    :param train_ratio: trainset person identities ratio
    :return:trainset and testset
    """
    folder1 = os.path.join(image_root, 'sequences', 'cam1')
    folder2 = os.path.join(image_root, 'sequences', 'cam2')
    dataset = {}
    for folder in os.listdir(folder1):
        pid = folder[-3:]
        dataset[pid] = {}
        image_folder = os.path.join(folder1, folder)
        dataset[pid]['1'] = [os.path.join(image_folder, image) for image in sorted(os.listdir(image_folder))
                             if image.endswith('.png')]
        image_folder = os.path.join(folder2, folder)
        dataset[pid]['2'] = [os.path.join(image_folder, image) for image in sorted(os.listdir(image_folder))
                             if image.endswith('.png')]
        pids = list(dataset.keys())
        random.shuffle(pids)
        num_train = int(len(pids) * train_ratio)

        trainset = {pids[i]: dataset[pids[i]] for i in range(num_train)}
        testset = {pids[i]: dataset[pids[i]] for i in range(num_train, len(pids))}
    return trainset, testset


def save_dataset(dataset, h5_file, pre_load=True):
    """
    save the dataset in .h5 file
    :param dataset: trainset or testset
    :param h5_file: filename to save
    :param pre_load: whether or not pre-load images
    :return:
    """
    with h5py.File(h5_file, 'w') as fw:
        for pid, cams in dataset.items():
            grp = fw.create_group(pid)
            for cam, tracklet in cams.items():
                if pre_load:
                    images = [np.array(Image.open(file)) for file in tracklet]
                    images = np.stack(images)
                    grp.create_dataset(cam, dtype=np.uint8, data=images)
                else:
                    images = [file.encode() for file in tracklet]
                    grp.create_dataset(cam, data=images)


def load_dataset(h5_file, pre_load=True):
    """
    load h5 file, return dataset in dict form
    dataset[pid][cam] = list of tracklets
    if pre-load is True tracklet is a list of Image array
    if pre-load is False tracklet consist of filenames
    :param h5_file: h5 file to load
    :param pre_load: whether of not pre-load
    :return: dataset in dict form
    """
    dataset = {}
    with h5py.File(h5_file, 'r') as fr:
        for pid, cams in fr.items():
            dataset[pid] = {}
            for cam, tracklet in cams.items():
                if pre_load:
                    images = np.array(tracklet, dtype=np.uint8)
                    dataset[pid][cam] = [[Image.fromarray(images[i]) for i in range(len(images))]]
                else:
                    images = [file.decode() for file in tracklet]
                    dataset[pid][cam] = [images]
    return dataset


def generate_split(image_root, data_root='data', num_splits=1, train_ratio=0.5, pre_load=True):
    """
    generate split and store dataset inot h5 file
    :param image_root: image root of ilids dataset
    :param data_root:  data root to store h5 file
    :param num_splits:  number of train-test splits
    :param train_ratio: trainset person identities ratio
    :param pre_load: whether or not pre-load image
    :return:
    """
    if not os.path.exists(data_root):
        os.mkdir(data_root)
        
    for i in range(num_splits):
        print('Generating {}/{} splits of ilids dataset'.format(i+1, num_splits))
        train_h5 = os.path.join(data_root, 'train_{}.h5'.format(i))
        test_h5 = os.path.join(data_root, 'test_{}.h5'.format(i))
        if num_splits == 1:
            train_h5 = os.path.join(data_root, 'train.h5')
            test_h5 = os.path.join(data_root, 'test.h5')
        trainset, testset = train_test(image_root, train_ratio)
        save_dataset(trainset, train_h5, pre_load)
        save_dataset(testset, test_h5, pre_load)


def ILIDS(h5_file=None, image_root=None, data_root='data', train_ratio=0.5, pre_load=True,
          training=True, **kwargs):
    """
    :param h5_file: h5 file that store the dataset
    :param image_root: if h5_file is None, use image root to generate dataset
    :param data_root:  data root to store train test split dataset
    :param train_ratio: trainset ratio
    :param pre_load: whether or not to load images
    :param training:  whether or not in training stage
    :return: dict form dataset
    """
    if h5_file is not None:
        return load_dataset(h5_file, pre_load)

    elif image_root is not None:
        if training:
            generate_split(image_root, data_root, 1, train_ratio, pre_load)
            return load_dataset(os.path.join(data_root, 'train.h5'), pre_load)
        else:
            raise NameError('Only in training stage can use `image_root` to load dataset')
    else:
        raise NameError('`image_root` and `h5_file` cannot both be None')
