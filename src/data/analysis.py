import argparse
from collections import defaultdict

import numpy as np
from tabulate import tabulate, SEPARATING_LINE

from label import Label
from myio import load_labels
from util import get_species_list
from data.crickets.ensifera_mingkai.split import test_files

from plotting import example_stats


MS = 1000.


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
        species_labels, label_files = load_labels(root_dir, species)
        row.append(len(label_files))
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
    # print(formatted_table)
    if save_file:
        with open(save_file, 'w') as o:
            o.write(formatted_table)
    return all_labels


def get_total_time(label_list):
    time = sum(t2 - t1 for (t1, t2), _ in label_list)
    return time

def main():
    args = get_args()
    root_dir = args.labels_directory
    species_list = get_species_list(root_dir)
    all_labels = load_and_tabulate_data(root_dir, species_list)
    for species, labels in all_labels.items():
        total_time = dict()
        split_time = dict(test=0., train=0.)
        print(species)
        for label, examples in labels.items():
            if label in Label.non_noise():
                for (t1, t2), filename in examples:
                    filename = filename.rstrip('.txt')
                    if filename not in total_time:
                        total_time[filename] = 0.
                    time = t2 - t1
                    total_time[filename] += time
                    split = 'test' if filename in test_files[species] else 'train'
                    split_time[split] += time
        total = sum(total_time.values())
        for fn, time in sorted(total_time.items(), key=lambda x: x[1], reverse=True):
            print(f"\t{fn}: {time} ({time / total * 100}%)")
        print(f"Train: {split_time['train'] / total}, Test: {split_time['test'] / total}")
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
