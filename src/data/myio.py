import os
from collections import defaultdict
from glob import glob

from label import Label


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


def load_labels(root_dir, species):
    species_labels = defaultdict(list)
    label_files = glob(os.path.join(root_dir, species, '*.txt'))
    for f in label_files:
        labels = load_label_file(f)
        for call_type, examples in labels.items():
            species_labels[call_type].extend(examples)
    return species_labels, label_files
