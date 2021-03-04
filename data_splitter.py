from typing import List
import os
import argparse
import random
random.seed(47)

parser = argparse.ArgumentParser(description="Split data filespath into Train/Val/Test")
parser.add_argument("-f", "--file", default="./all.txt", help="filespath path. Ex: all.txt")
parser.add_argument("-o", "--out_folder", default="./", help="filespath path. Ex: all.txt")

def split_filespath(filespath:List[str], majority_percentage:float = 0.9):
    class_data = {}
    for file in filespath:
        cls_name = file.split(os.sep)[-2]
        if cls_name in class_data:
            class_data[cls_name].append(file)
        else:
            class_data[cls_name] = [file]

    majority = []
    minority = []
    for cls_name, files in class_data.items():
        stop = int(majority_percentage*len(files))
        majority.extend(files[:stop])
        minority.extend(files[stop:])

    random.shuffle(majority)
    random.shuffle(minority)
    return majority, minority


if __name__ == "__main__":
    args = parser.parse_args()
    
    files = []
    with open(args.file, 'r') as reader:
        files = reader.read().splitlines()
    
    #Train/Val
    train, val = split_filespath(files, 0.8)
    #Val/Test
    val, test = split_filespath(val, 0.5)

    for files, outfile in [(train, 'train-baseline.txt'), (val, 'val-baseline.txt'), (test, 'test-baseline.txt')]:
        outfile = os.path.join(args.out_folder, outfile)
        with open(outfile, 'w') as writer:
            for file in files:
                writer.write(file+"\n")
