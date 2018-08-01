import h5py
import numpy as np
from PIL import Image


def save_dataset(dataset, h5_file, image_root=None, of_root=None, pre_load=True):
    """
    save the dataset in .h5 file
    :param dataset: trainset or testset
    :param h5_file: filename to save
    :param image_root: image root of ilids dataset
    :param of_root: optical flow of ilids dataset
    :param pre_load: whether or not pre-load images
    :return:
    """
    with h5py.File(h5_file, 'w') as fw:
        for pid, cams in dataset.items():
            grp = fw.create_group(pid)
            for cam, tracklet in cams.items():
                if pre_load:
                    images = np.stack(np.array(Image.open(file)) for file in tracklet)
                    if of_root is not None:
                        ofs = np.stack(np.array(Image.open(file.replace(image_root, of_root)))[:, :, :2]
                                       for file in tracklet)
                        images = np.concatenate([images, ofs], axis=-1)
                    grp.create_dataset(cam, dtype=np.uint8, data=images)
                else:
                    images = [file.encode() for file in tracklet]
                    grp.create_dataset(cam, data=images)


def load_dataset(h5_file):
    """
    load h5 file, return dataset in dict form
    dataset[pid][cam] = list of tracklets
    if pre-load is True tracklet is a list of Image array
    if pre-load is False tracklet consist of filenames
    :param h5_file: h5 file to load
    :return: dataset in dict form
    """
    dataset = {}
    with h5py.File(h5_file, 'r') as fr:
        for pid, cams in fr.items():
            dataset[pid] = {}
            for cam, tracklet in cams.items():
                try:
                    images = [file.decode() for file in tracklet]
                    dataset[pid][cam] = [images]
                except:
                    images = np.array(tracklet, dtype=np.uint8)
                    dataset[pid][cam] = [[image for image in images]]
    return dataset
