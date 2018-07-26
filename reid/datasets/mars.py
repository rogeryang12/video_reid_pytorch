import os


def MARS(image_root=None, training=True, **kwargs):
    """
    :param image_root: image root to generate dataset
    :param training: whether or not in training stage
    :return: dataset in dict form
    """
    image_root = os.path.join(image_root, 'bbox_train' if training else 'bbox_test')
    dataset = {}
    pids = sorted(os.listdir(image_root))
    for pid in pids:
        dataset[pid] = {}
        folder = os.path.join(image_root, pid)
        files = [file for file in sorted(os.listdir(folder)) if file.endswith('jpg')]
        index = [0] + [i for i in range(1, len(files)) if files[i][7:11] != files[i - 1][7:11]] + [None]
        files = [os.path.join(folder, file) for file in files]
        for i in range(len(index) - 1):
            cam = os.path.basename(files[index[i]])[5]
            if cam not in dataset[pid]:
                dataset[pid][cam] = []
            dataset[pid][cam].append(files[index[i]:index[i + 1]])
    return dataset
