import argparse
import json
from pathlib import Path
import os
import glob
import pandas as pd
import h5py
import csv
import shutil

parser = argparse.ArgumentParser(description='Parallel') 
parser.add_argument('--path2dir', type=str, default='./data')
parser.add_argument('--rate', type=float, default=0.2)
parser.add_argument('--category', type=int, default=1000)
parser.add_argument('--thread', type=int, default=40)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

dir_path = os.path.join(args.path2dir, 'csv_rate' + str(args.rate) + '_category' + str(args.category))

category_per_thread = args.category / args.thread
for i in range(0, args.category):
    
    dir_name = "csv" + str(int(i/category_per_thread))
    if not os.path.isdir(os.path.join(dir_path, dir_name)):
        os.mkdir(os.path.join(dir_path, dir_name))
    file_name = str(i).zfill(5) + ".csv"
    shutil.move(os.path.join(dir_path,file_name), os.path.join(dir_path,dir_name,file_name))
    print(os.path.join(dir_path,dir_name,file_name))
