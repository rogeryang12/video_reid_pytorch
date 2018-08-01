import torch
from torch import optim
from argparse import ArgumentParser

from reid.models import CNN
from reid.datasets import TrainLabel
from reid.losses import CrossEntropyLoss
from reid.trainer import Trainer
import reid.transforms as T


def main():

    parser = ArgumentParser(description='Train a Reid network')
    parser.add_argument('--experiment_root', default='CNN')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--cnn', default='resnet50', choices=['inception3', 'resnet50'])
    parser.add_argument('--ckpt', default='models/resnet50.pth')
    parser.add_argument('--emb_dim', default=1024, type=int)

    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--gamma', default=0.1, type=float)

    parser.add_argument('--of_root', default=None)

    parser.add_argument('--dataset', default='mars', choices=['ilids', 'mars'])
    parser.add_argument('--image_root', default='root_of_mars')
    parser.add_argument('--iters', default=16000, type=int)
    parser.add_argument('--save_iter', default=1000, type=int)
    parser.add_argument('--lr_decay_iter', default=5000, type=int)

    # parser.add_argument('--dataset', default='ilids', choices=['ilids', 'mars'])
    # parser.add_argument('--h5_file', default=None)
    # parser.add_argument('--image_root', default='root_of_ilids')
    # parser.add_argument('--pre_load', default=True)
    # parser.add_argument('--iters', default=1000, type=int)
    # parser.add_argument('--save_iter', default=100, type=int)
    # parser.add_argument('--lr_decay_iter', default=500, type=int)

    args = parser.parse_args()
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3,0'

    transform = [T.Resize((288, 144)), T.RandomCrop((256, 128)), T.RandomHorizontalFlip()]
    dataset = TrainLabel(**vars(args), transform=transform)
    model = CNN(args.emb_dim, len(dataset.pids), args.cnn, ckpt=args.ckpt)
    optimizer = optim.SGD(model.parameters(), args.lr, args.momentum, weight_decay=args.weight_decay)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    celoss = CrossEntropyLoss()

    trainer = Trainer(args, dataset, model, optimizer, device)

    def cal_loss(data, model):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        embs, logits = model(images)
        loss, prec = celoss(logits, labels)
        log = ['loss', loss.item(), 'prec', prec]
        return loss, log

    trainer.train(cal_loss)


if __name__ == '__main__':
    main()

