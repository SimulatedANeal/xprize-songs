import argparse
import os
from collections import defaultdict
from glob import glob

import numpy as np

from label import Label


MS = 1000.


def load_label_file(filepath):
    label_dict = defaultdict(list)
    filename = os.path.basename(filepath)
    with open(filepath, 'r') as f:
        for line in f:
            try:
                start, stop, raw_label = line.strip().split('\t')
            except ValueError:
                print(f"Invalid line '{line.strip()}' in file {filepath}")
            try:
                label = Label(int(raw_label))
            except ValueError:
                print(
                    f"Unexpected label '{raw_label}' in file {filepath}. "
                    f"Treating as {Label.NOISE}.")
                label = Label.NOISE
            label_dict[label].append(((float(start), float(stop)), filename))
        # # Detect singletons
        # for k, v in label_dict.items():
        #     if len(v) == 1:
        #         print(f"Singleton {k} in {filepath}")
    return label_dict


def show_label_stats(label_dict):
    for label, times in label_dict.items():
        print(f"{label.name} stats:" )
        print(f"\tNumber of labelled examples: {len(times)}")
        lengths_ms = [(stop - start) * MS for (start, stop), _ in times]
        print(
            f"\tMaximum length: {max(lengths_ms):.2f}ms "
            f"(File: {times[lengths_ms.index(max(lengths_ms))][1]})")
        print(
            f"\tMinimum length: {min(lengths_ms):.2f}ms "
            f"(File: {times[lengths_ms.index(min(lengths_ms))][1]})")
        print(f"\tMean length: {np.mean(lengths_ms):.2f}ms")



def main():
    args = get_args()
    root_dir = args.labels_directory

    species_list = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))]
    species_list.sort()

    for species in species_list:
        all_labels  = defaultdict(list)
        print(f"{species.upper()}\n{'=' * len(species)}")
        label_files = glob(os.path.join(root_dir, species, '*.txt'))
        print(f"Number of labelled audio files: {len(label_files)}")
        for f in label_files:
            labels = load_label_file(f)
            for k in labels:
                all_labels[k].extend(labels[k])
        show_label_stats(all_labels)
        print("")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-directory", "-d", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
