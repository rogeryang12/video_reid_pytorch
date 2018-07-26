import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .ilids import ILIDS
from .mars import MARS


_factory = {
    'ilids': ILIDS,
    'mars': MARS,
}


class TrainBase(Dataset):

    def __init__(self, dataset='ilids', transform=None, h5_file=None, image_root=None, data_root='data',
                 train_ratio=0.5, pre_load=True, **kwargs):
        if dataset not in _factory.keys():
            raise NotImplementedError('`{}` dataset not supported yet'.format(dataset))
        self.dataset = _factory[dataset](h5_file=h5_file, image_root=image_root, data_root=data_root,
                                         train_ratio=train_ratio, pre_load=pre_load, training=True)

        self.pids = sorted(list(self.dataset.keys()))
        self.label_dict = {pid: label for label, pid in enumerate(self.pids)}
        self.pre_load = dataset == 'ilids' and pre_load

        trans = [T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        self.trans = T.Compose(transform + trans if transform is not None else trans)

    def transform(self, img):
        if not self.pre_load:
            img = Image.open(img)
        return self.trans(img)

    def transform_list(self, img_list):
        imgs = [self.transform(img) for img in img_list]
        return torch.stack(imgs)


class TrainLabel(TrainBase):
    """
    return a image and its label
    """

    def __init__(self, **kwargs):
        super(TrainLabel, self).__init__(**kwargs)
        self.images = []
        self.labels = []
        for pid in self.pids:
            cams = sorted(self.dataset[pid].keys())
            for cam in cams:
                for track in self.dataset[pid][cam]:
                    self.labels.extend([self.label_dict[pid]] * len(track))
                    self.images.extend(track)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.transform(self.images[item]), self.labels[item]


class TrainTIBatch(TrainBase):
    """
    track_tensor: track_num x image_num x c x h x w     track_num sequences each consists of image_num images
    labels: track_num x image_num    person label of track_num x image_num images
    """
    def __init__(self, track_num, image_num, **kwargs):
        super(TrainTIBatch, self).__init__(**kwargs)
        self.track_num = track_num
        self.image_num = image_num
        self.track_dataset = {pid: [] for pid in self.pids}
        self.generate_tracks()

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, item):
        pid = self.pids[item]
        tracks = random.sample(self.track_dataset[pid], self.track_num)
        track_tensor = [self.transform_list(track) for track in tracks]
        track_tensor = torch.stack(track_tensor)
        return track_tensor, torch.LongTensor([[self.label_dict[pid]] * self.image_num] * self.track_num)

    def generate_tracks(self):
        for pid in self.pids:
            pid_tracks = self.track_dataset[pid]
            cams = sorted(self.dataset[pid].keys())
            for cam in cams:
                for track in self.dataset[pid][cam]:
                    track_len = len(track)
                    if track_len < self.image_num:
                        track.extend([track[-1]] * (self.image_num - track_len))
                        track_len = self.image_num
                    for i in range(track_len - self.image_num + 1):
                        pid_tracks.append(track[i: i+self.image_num])

            diff = self.track_num - len(pid_tracks)
            if diff > 0:
                pid_tracks.extend(pid_tracks[i % len(pid_tracks)] for i in range(diff))


class TrainPairImages(TrainTIBatch):
    """
    track_tensor: 2 x image_num x c x h x w     2 sequences each consists of image_num images
    labels: 2 x image_num    person label of 2 x image_num images
    same: True or False      whether 2 sequences are the same person
    """

    def __init__(self, image_num, neg_pos_ratio=1, **kwargs):
        """
        :param neg_pos_ratio: number of negative pairs and positive pairs ratio
        """
        super(TrainPairImages, self).__init__(2, image_num, **kwargs)
        self.neg_pos_ratio = neg_pos_ratio

    def __len__(self):
        return len(self.pids) * (1 + self.neg_pos_ratio)

    def __getitem__(self, item):
        if item < len(self.pids):
            pid1 = pid2 = pid = self.pids[item]
            tracks = random.sample(self.track_dataset[pid], 2)
            same = True
        else:
            pid1, pid2 = random.sample(self.pids, 2)
            tracks = [random.choice(self.track_dataset[pid1]),
                      random.choice(self.track_dataset[pid2])]
            same = False

        track_tensor = torch.stack([self.transform_list(tracks[0]), self.transform_list(tracks[1])])
        labels = torch.LongTensor([[self.label_dict[pid1]] * self.image_num,
                                   [self.label_dict[pid2]] * self.image_num])
        return track_tensor, labels, same


class TestBase(Dataset):

    def __init__(self, dataset='ilids', h5_file=None, image_root=None, data_root='data',
                 train_ratio=0.5, pre_load=True, resize=None, crop_size=None, flip=False, **kwargs):
        if dataset not in _factory.keys():
            raise NotImplementedError('`{}` dataset not supported yet'.format(dataset))
        self.dataset = _factory[dataset](h5_file=h5_file, image_root=image_root, data_root=data_root,
                                         train_ratio=train_ratio, pre_load=pre_load, training=False)

        self.pre_load = dataset == 'ilids' and pre_load
        self.resize = resize
        self.crop_size = crop_size
        self.flip = flip
        self.totensor = T.Compose([T.ToTensor(),
                                   T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def transform(self, img):
        if not self.pre_load:
            img = Image.open(img)
        if self.resize is not None:
            img = T.Resize(self.resize)(img)
        if self.crop_size is not None:
            imgs = T.FiveCrop(self.crop_size)(img)
        else:
            imgs = [img]
        if self.flip:
            imgs = [T.RandomHorizontalFlip(p=1)(img) for img in imgs] + list(imgs)
        imgs = torch.stack([self.totensor(img) for img in imgs])
        return imgs


class TestImages(TestBase):
    """
    return image and its augment in shape au*c*h*w
    index shows the tracklet splits
    pids and cams are the person id and camera id of the tracklet
    query and gallery determine the query gallery split
    """
    def __init__(self, **kwargs):
        super(TestImages, self).__init__(**kwargs)
        self.images, self.pids, self.cams = [], [], []
        self.query, self.gallery = [], []
        self.index = [0]
        self.get_query_gallery(kwargs.get('dataset'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.transform(self.images[item])

    def get_query_gallery(self, dataset):
        if dataset == 'mars':
            del self.dataset['00-1']

        track_num = 0
        for pid in sorted(self.dataset.keys()):
            cams = sorted(self.dataset[pid].keys())
            for cam in cams:
                tracks = self.dataset[pid][cam]
                num = len(tracks)
                for track in tracks:
                    self.images.extend(track)
                    self.index.append(len(self.images))
                self.pids.extend([int(pid)] * num)
                self.cams.extend([int(cam)] * num)
                track_num += num
                if dataset == 'mars' and pid != '0000' and len(cams) > 1:
                    self.query.append(track_num - 1)
                if dataset == 'ilids' and cam == '1':
                    self.query.append(track_num - 1)
                if dataset == 'ilids' and cam == '2':
                    self.gallery.append(track_num - 1)
        if dataset == 'mars':
            self.gallery = list(range(track_num))
        self.query = np.array(self.query)
        self.gallery = np.array(self.gallery)
        self.pids = np.array(self.pids)
        self.cams = np.array(self.cams)

    # def get_query_gallery(self, dataset):
    #     if dataset == 'mars':
    #         del self.dataset['00-1']
    #         # self.index.append(0)
    #         tracklet_num = 0
    #         for pid, cams in self.dataset.items():
    #             for cam, tracklets in cams.items():
    #                 num = len(tracklets)
    #                 for tracklet in tracklets:
    #                     self.images.extend(tracklet)
    #                     self.index.append(len(self.images))
    #                 self.pids.extend([int(pid)] * num)
    #                 self.cams.extend([int(cam)] * num)
    #                 tracklet_num += num
    #                 if pid != '0000' and len(cams) > 1:
    #                     self.query.append(tracklet_num - 1)
    #         for pid in sorted(self.dataset.keys()):
    #             cams = sorted(self.dataset[pid].keys())
    #             for cam in cams:
    #                 num = len(self.dataset[pid][cam])
    #                 for tracklet in self.dataset[pid][cam]:
    #                     self.images.extend(tracklet)
    #                     self.index.append(len(self.images))
    #                 self.pids.extend([int(pid)] * num)
    #                 self.cams.extend([int(cam)] * num)
    #                 tracklet_num += num
    #                 if pid != '0000' and len(cams) > 1:
    #                     self.query.append(tracklet_num - 1)
    #         self.gallery = list(range(tracklet_num))
    #     elif dataset == 'ilids':
    #         tracklet_num = 0
    #         for pid, cams in self.dataset.items():
    #             for cam, tracklets in cams.items():
    #                 self.images.extend(tracklets[0])
    #                 self.index.append(len(self.images))
    #                 self.pids.append(int(pid))
    #                 self.cams.append(int(cam))
    #                 if cam == '1':
    #                     self.query.append(tracklet_num)
    #                 else:
    #                     self.gallery.append(tracklet_num)
    #                 tracklet_num += 1
    #     self.query = np.array(self.query)
    #     self.gallery = np.array(self.gallery)
    #     self.pids = np.array(self.pids)
    #     self.cams = np.array(self.cams)

