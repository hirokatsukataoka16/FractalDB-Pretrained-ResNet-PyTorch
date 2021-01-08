# -*- coding: utf-8 -*-

import argparse

def conf():
    parser = argparse.ArgumentParser(description="PyTorch FractalDB Pre-training")
    # model name
    parser.add_argument("--dataset", default="FractalDB-1000", type = str, help="model name")
    # paths
    parser.add_argument("--path2traindb", default="./data/FractalDB-1000", type = str, help="path to FractalDB")
    parser.add_argument("--path2valdb", default="./data/FractalDB-1000", type = str, help="path to FractalDB (basically, there is no validation set on FractalDB)")
    parser.add_argument("--path2weight", default="./data/weight", type = str, help="path to trained weight")
    parser.add_argument('--val', default=False, action='store_true', help='If true, training is not performed.')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    # network settings
    parser.add_argument("--usenet", default="resnet50", type = str, help="use network")
    parser.add_argument("--epochs", default=90, type = int, help="end epoch")
    parser.add_argument("--numof_classes", default=1000, type = int, help="num of classes")
    # model hyper-parameters
    parser.add_argument("--lr", default=0.1, type = float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type = float, help="momentum")
    parser.add_argument("--weight_decay", default=1e-4, type = float, help="weight decay")
    # etc
    parser.add_argument("--start-epoch", default=1, type = int, help="input batch size for training")
    parser.add_argument("--batch_size", default=256, type = int, help="input batch size for training")
    parser.add_argument("--val-batch_size", default=256, type=int, help="input batch size for testing")
    parser.add_argument("--img_size", default=256, type = int, help="image size")
    parser.add_argument("--crop_size", default=224, type = int, help="crop size")
    parser.add_argument('--no_multigpu', default=False, action='store_true', help='If true, training is not performed.')
    parser.add_argument("--no-cuda", default=False, action="store_true", help="disables CUDA training")
    parser.add_argument("--gpu_id", default=-1, type = int, help="gpu id")
    parser.add_argument("--num_workers", default=8, type = int, help="num of workers (data_loader)")
    parser.add_argument("--save-interval", default=10, type = int, help="save every N epoch")
    parser.add_argument("--log-interval", default=100, type=int, help="how many batches to wait before logging training status")
    parser.add_argument("--seed", default=1, type=int, help="seed")
    args = parser.parse_args()
    return args
