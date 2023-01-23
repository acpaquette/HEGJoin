#!/usr/bin/env python

import os
import argparse
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('file_folder', type=str, default='',
                        help='path to the folder holding output files')
    parser.add_argument('file_prefix', type=str, default='',
                        help='path to the folder holding output files')

    return parser.parse_args()

def get_times(file_folder, file_prefix):
    files = glob.glob(os.path.join(file_folder, file_prefix) + '*')
    files = sorted(files, key = (lambda x : int(os.path.splitext(x)[0].split('_')[-1])))
    time_sets = []
    for file in files:
        cpu_num = os.path.splitext(file.split("/")[-1].split("_")[-1])[0]
        with open(file, 'r') as fp:
            data = fp.readlines()
            try:
                epsilon = data[6].split(" ")[-1].rstrip()
                dimension = data[7][-2]
                total_time_line = data[-5]
                total_time = total_time_line.split(" ")[-1].rstrip()
                gpu_time_line = data[-3]
                gpu_time = gpu_time_line.split(" ")[-1].rstrip()
                cpu_time_line = data[-2]
                cpu_time = cpu_time_line.split(":")[1].split(" ")[1].rstrip()
                ratio = data[-1].split(":")[-1].replace(")", "").replace(" ", "").rstrip()
                time_sets.append((dimension, epsilon, cpu_num, total_time, gpu_time, cpu_time, ratio))
            except Exception as e:
                print("Failed on {}".format(cpu_num))
    return time_sets

def main(args):
    time_sets = get_times(args.file_folder, args.file_prefix)
    for time_set in time_sets:
        print(f"{time_set[0]}, {time_set[1]}, {time_set[2]}, {time_set[3]}, {time_set[4]}, {time_set[5]}, {time_set[6]}")


if __name__ == '__main__':
    main(parse_args())
