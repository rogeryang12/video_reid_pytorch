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
        track_tensor = []
        for track in tracks:
            track_tensor.append(torch.stack([self.transform(img) for img in track]))
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


class TrainKImagePair(TrainBase):
    """
    seq: 2 x image_num x c x h x w     2 sequences each consists of image_num images
    labels: 2 x image_num    person label of 2 x image_num images
    same: True or False      whether 2 sequences are the same person
    """

    def __init__(self, image_num=8, sequential=True, neg_pos_ratio=0, **kwargs):
        """
        :param image_num: number of images to select from a tracklet
        :param sequential:  whether or not images in sequential order
        :param neg_pos_ratio:  negative pair and positive pair ratio
        """
        super(TrainKImagePair, self).__init__(**kwargs)
        self.image_num = image_num
        self.sequential = sequential
        self.neg_pos_ratio = neg_pos_ratio
        self.generate_tracklets()

    def __len__(self):
        return len(self.pids) * (self.neg_pos_ratio + 1)

    def __getitem__(self, item):
        if item < len(self.pids):
            same = True
            pid1 = pid2 = self.pids[item]
            tracklets1 = tracklets2 = self.tracklet_dataset[pid1]
            if len(tracklets1) < 2:
                index1, index2 = 0, 0
            else:
                index1, index2 = random.sample(range(len(tracklets1)), 2)
        else:
            same = False
            pid1, pid2 = random.sample(self.pids, 2)
            tracklets1 = self.tracklet_dataset[pid1]
            tracklets2 = self.tracklet_dataset[pid2]
            index1 = random.choice(range(len(tracklets1)))
            index2 = random.choice(range(len(tracklets2)))
        track1 = tracklets1[index1]
        track2 = tracklets2[index2]
        if self.sequential:
            x = random.randint(0, len(track1) - self.image_num)
            seq1 = track1[x: x + self.image_num]
            x = random.randint(0, len(track2) - self.image_num)
            seq2 = track2[x: x + self.image_num]
        else:
            seq1 = random.sample(track1, self.image_num)
            seq2 = random.sample(track2, self.image_num)
        seq1 = torch.stack([self.transform(img) for img in seq1])
        seq2 = torch.stack([self.transform(img) for img in seq2])
        labels = [[self.label_dict[pid1]] * self.image_num,
                  [self.label_dict[pid2]] * self.image_num]

        return torch.stack([seq1, seq2]), torch.LongTensor(labels), same

    def generate_tracklets(self):
        self.tracklet_dataset = {}
        for pid, cams in self.dataset.items():
            self.tracklet_dataset[pid] = []
            for cam, tracklets in cams.items():
                for tracklet in tracklets:
                    if len(tracklet) < self.image_num:
                        tracklet = tracklet + [tracklet[-1]] * (self.image_num - len(tracklet))
                    self.tracklet_dataset[pid].append(tracklet)
        del self.dataset


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


# class KImage(Dataset):
#
#     def __init__(self, train_h5, image_num, transform=None, shuffle=True):
#         self.dataset = load_dataset(train_h5)
#         self.pids = sorted(list(self.dataset.keys()))
#         self.label_dict = {pid: label for label, pid in enumerate(self.pids)}
#         self.image_num = image_num
#         self.transform = transform
#         self.shuffle = shuffle
#
#     def __len__(self):
#         return len(self.pids)
#
#     def __getitem__(self, item):
#         pid = self.pids[item]
#         cam1, cam2 = random.sample(['1', '2'], 2)
#
#         images1 = self.dataset[pid][cam1]
#         images2 = self.dataset[pid][cam2]
#         if self.shuffle is True:
#             seq1 = random.sample(images1, self.image_num)
#             seq2 = random.sample(images2, self.image_num)
#         else:
#             x = random.randint(0, len(images1) - self.image_num)
#             seq1 = images1[x: x + self.image_num]
#             x = random.randint(0, len(images2) - self.image_num)
#             seq2 = images2[x: x + self.image_num]
#         if self.transform is not None:
#             seq1 = torch.stack([self.transform(img) for img in seq1])
#             seq2 = torch.stack([self.transform(img) for img in seq2])
#
#             seq = torch.stack([seq1, seq2])
#         else:
#             seq = seq1 + seq2
#         labels = torch.LongTensor([self.label_dict[pid]] * 2 * self.image_num)
#
#         return seq, labels
#
#
# class Test(Dataset):
#
#     def __init__(self, test_h5, size=None, crop_size=None, flip=False):
#         self.testset = load_dataset(test_h5)
#         self.pids = list(self.testset.keys())
#         self.size = size
#         self.crop_size = crop_size
#         self.flip = flip
#         self.totensor = transforms.Compose([transforms.ToTensor(),
#                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
#
#     def __len__(self):
#         return len(self.pids)
#
#     def __getitem__(self, item):
#         pid = self.pids[item]
#         seq1 = self.testset[pid]['1']
#         seq2 = self.testset[pid]['2']
#         seq1 = torch.stack([self.transform(img) for img in seq1])
#         seq2 = torch.stack([self.transform(img) for img in seq2])
#         return seq1.transpose(0, 1), seq2.transpose(1, 0), pid
#
#     def transform(self, img):
#         if self.size is not None:
#             img = transforms.Resize(self.size)(img)
#         if self.crop_size is not None:
#             imgs = transforms.FiveCrop(self.crop_size)(img)
#         else:
#             imgs = [img]
#         if self.flip:
#             imgs = [transforms.RandomHorizontalFlip(p=1)(img) for img in imgs] + list(imgs)
#         imgs = torch.stack([self.totensor(img) for img in imgs])
#         return imgs



