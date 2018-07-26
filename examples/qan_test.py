import os
from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.data import DataLoader

from reid.datasets import TestImages
from reid.models import QAN
from reid.evaluation import cal_cmc_aps


def main():
    parser = ArgumentParser(description='Evaluate a Reid network')

    parser.add_argument('--experiment_root', default='QAN')
    parser.add_argument('--dataset', default='mars', choices=['ilids', 'mars'])
    parser.add_argument('--image_root', default='root_of_mars')
    # parser.add_argument('--dataset', default='ilids', choices=['ilids', 'mars'])
    # parser.add_argument('--h5_file', default='data/test.h5')
    # parser.add_argument('--pre-load', default=True)

    parser.add_argument('--cnn', default='resnet50', choices=['inception3', 'resnet50'])
    parser.add_argument('--checkpoint', default='model20000.pkl')
    parser.add_argument('--emb_dim', default=1024, type=int)
    parser.add_argument('--batch_size', default=1536, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3'

    dataset = TestImages(**vars(args))  #, flip=True, resize=(256, 128) =(288, 144), crop_size
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QAN(args.emb_dim, cnn=args.cnn)
    model.load_state_dict(torch.load(os.path.join(args.experiment_root, args.checkpoint)), False)
    model = model.to(device).eval()
    model = torch.nn.DataParallel(model)

    torch.set_grad_enabled(False)

    embs = []
    quals = []
    images_num = len(dataset)
    for k, images in enumerate(loader, 1):
        images = images.to(device)
        b, au, c, h, w = images.size()
        emb, qual = model(images.view(-1, c, h, w))
        embs.append(emb.view(b, au, -1).mean(dim=1).cpu())
        quals.append(qual.view(b, au, -1).mean(dim=1).cpu())
        end = k * args.batch_size
        end = end if end < images_num else images_num
        print('\rEmbeding {}/{}'.format(end, images_num), flush=True, end=' ')
    print()

    torch.cuda.empty_cache()
    embs = torch.cat(embs)
    quals = torch.cat(quals)
    feats = []
    for i in range(len(dataset.index) - 1):
        s = slice(dataset.index[i], dataset.index[i + 1])
        feats.append((embs[s] * quals[s]).sum(dim=0) / quals[s].sum(dim=0))
    feats = torch.stack(feats).to(device)
    np.save(os.path.join(args.experiment_root, 'features.npy'), feats.cpu().numpy())

    # feats = np.load(os.path.join(args.experiment_root, 'features.npy'))
    # feats = torch.from_numpy(feats).to(device)
    log_file = os.path.join(args.experiment_root, 'eval.json')
    cal_aps = True if args.dataset == 'mars' else False
    batch_size = 10 if args.dataset == 'mars' else None
    cal_cmc_aps(feats, dataset.pids, dataset.cams, dataset.query, dataset.gallery, log_file, cal_aps, batch_size)


if __name__ == '__main__':
    main()

