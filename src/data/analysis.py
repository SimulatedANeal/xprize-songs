import argparse
import os
from collections import defaultdict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate, SEPARATING_LINE


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


def load_and_tabulate_data(root_dir, species_list, save_file='table.txt'):
    table = []
    header = ["Species", "# Labelled Audio Files"]
    for label in Label.valid():
        header.extend([
            f"# {label.name} Examples",
            f"{label.name} Max. length (ms)",
            f"{label.name} Min. length (ms)",
            f"{label.name} Mean length (ms)"])

    all_labels = dict()
    lengths = defaultdict(list)
    for species in species_list:
        row = [species]
        species_labels = defaultdict(list)
        label_files = glob(os.path.join(root_dir, species, '*.txt'))
        row.append(len(label_files))
        for f in label_files:
            labels = load_label_file(f)
            for call_type, examples in labels.items():
                species_labels[call_type].extend(examples)
        all_labels[species] = species_labels
        for label in Label.valid():
            examples = species_labels[label]
            if len(examples):
                row.append(len(examples))
                lengths_ms = [(t2 - t1) * MS for (t1, t2), _ in examples]
                row.append(float(f"{max(lengths_ms):.2f}"))
                row.append(float(f"{min(lengths_ms):.2f}"))
                row.append(float(f"{np.mean(lengths_ms):.2f}"))
                lengths[label].extend(lengths_ms)
            else:
                row.extend(["", "", "", ""])
        table.append(row)
    total_row = ["All Data"]
    total_row.append(sum(r[1] for r in table))  # File count
    for label in Label.valid():
        total_row.append(len(lengths[label]))
        total_row.append(float(f"{max(lengths[label]):.2f}"))
        total_row.append(float(f"{min(lengths[label]):.2f}"))
        total_row.append(float(f"{np.mean(lengths[label]):.2f}"))
    table.append(SEPARATING_LINE)
    table.append(total_row)
    formatted_table = tabulate(
        table,
        headers=header,
        tablefmt="latex",
        colalign=['left'] + ['decimal'] * (len(table[0]) - 1),
        floatfmt=".2f")
    print(formatted_table)
    if save_file:
        with open(save_file, 'w') as o:
            o.write(formatted_table)
    return all_labels


def plot_example_stats(all_species_dict, title, bar_fn=len, y_label='Count'):
    labels = list(all_species_dict.keys())
    width = 0.35  # the width of the bars
    total_height = [0] * len(labels)
    fig, ax = plt.subplots()
    for i, l in enumerate(Label.non_noise()):
        xx = [bar_fn(v[l]) for k,v in all_species_dict.items()]
        ax.bar(labels, xx, width, label=l.name, bottom=total_height)
        total_height = [x + h for (x, h) in zip(xx, total_height)]
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    ax.legend()
    fig.tight_layout()
    plt.show()


def get_total_time(label_list):
    time = sum(t2 - t1 for (t1, t2), _ in label_list)
    return time

def main():
    args = get_args()
    root_dir = args.labels_directory

    species_list = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))]
    species_list.sort()

    all_labels = load_and_tabulate_data(root_dir, species_list)
    # plot_example_stats(
    #     all_species_dict=all_labels,
    #     title='Example Count by Species and Type',
    #     bar_fn=len,
    #     y_label='Count')
    # plot_example_stats(
    #     all_species_dict=all_labels,
    #     title='Total Example Time by Species and Type',
    #     bar_fn=get_total_time,
    #     y_label="Time (s)")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-directory", "-d", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
