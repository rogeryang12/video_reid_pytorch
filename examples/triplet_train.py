import torch
from torch import optim
import torchvision.transforms as T
from argparse import ArgumentParser

from reid.models import CNN
from reid.datasets import TrainTIBatch
from reid.losses import CrossEntropyLoss, TripletLoss
from reid.trainer import Trainer


def main():

    parser = ArgumentParser(description='Train a Reid network')
    parser.add_argument('--experiment_root', default='Triplet')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--batch_size', default=18, type=int)
    parser.add_argument('--track_num', default=4, type=int)
    parser.add_argument('--image_num', default=1, type=int)

    parser.add_argument('--cnn', default='resnet50', choices=['inception3', 'resnet50'])
    parser.add_argument('--ckpt', default='models/resnet50.pth')
    parser.add_argument('--emb_dim', default=1024, type=int)
    parser.add_argument('--margin', default='soft')
    parser.add_argument('--mode', default='adaptive')

    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--gamma', default=0.1, type=float)

    parser.add_argument('--dataset', default='mars', choices=['ilids', 'mars'])
    parser.add_argument('--image_root', default='/data1/lixz/mars')
    parser.add_argument('--iters', default=30000, type=int)
    parser.add_argument('--save_iter', default=1000, type=int)
    parser.add_argument('--lr_decay_iter', default=10000, type=int)

    # parser.add_argument('--dataset', default='ilids', choices=['ilids', 'mars'])
    # parser.add_argument('--data_root', default='data')
    # parser.add_argument('--h5_file', default=None)
    # parser.add_argument('--image_root', default='/data1/lixz/ilids')
    # parser.add_argument('--pre_load', default=True)
    # parser.add_argument('--iters', default=2000, type=int)
    # parser.add_argument('--save_iter', default=100, type=int)
    # parser.add_argument('--lr_decay_iter', default=1000, type=int)

    args = parser.parse_args()
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3,2'

    transform = [T.Resize((288, 144)), T.RandomCrop((256, 128)), T.RandomHorizontalFlip()]
    dataset = TrainTIBatch(**vars(args), transform=transform)
    model = CNN(args.emb_dim, len(dataset.pids), args.cnn, ckpt=args.ckpt)
    optimizer = optim.SGD(model.parameters(), args.lr, args.momentum, weight_decay=args.weight_decay)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    celoss = CrossEntropyLoss()
    triloss = TripletLoss(margin=args.margin, mode=args.mode).to(device)

    trainer = Trainer(args, dataset, model, optimizer, device)

    def cal_loss(data, model):
        images, labels = data
        b, k, i, c, h, w = images.size()
        images, labels = images.view(-1, c, h, w).to(device), labels.view(-1).to(device)
        embs, logits = model(images)
        loss1, prec = celoss(logits, labels)
        loss2, top1 = triloss(embs, labels)
        loss = loss1 + loss2
        log = ['loss', loss.item(), 'classloss', loss1.item(), 'triloss', loss2.item(), 'top1', top1, 'prec', prec]
        return loss, log

    trainer.train(cal_loss)


if __name__ == '__main__':
    main()

