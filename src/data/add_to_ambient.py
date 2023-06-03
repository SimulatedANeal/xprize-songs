import argparse
import csv
import os
import random

import tensorflow as tf

from src.data.myio import get_wav_file
from src.data.preprocessing import pad_valid_segments


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exported-reviews", required=True)
    parser.add_argument("--directory-dataset", required=True)
    return parser.parse_args()


def main():
    args = get_args()
    dataset_dir = args.directory_dataset
    window, sample_rate = os.path.basename(dataset_dir).split('_')
    window_ms = int(window[:-2])
    sample_rate =int(sample_rate[:-3]) * 1000
    with open(args.exported_reviews, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['accepted'] == 'False':
                subset = 'train' if random.random() <= 0.8 else 'test'
                save_dir = os.path.join(dataset_dir, subset, 'ambient')
                filename = os.path.basename(row['source_file'])[:-4]
                audio, sr, length_s = get_wav_file(row['source_file'], sample_rate)
                sr = sr.numpy()
                segments = pad_valid_segments(
                    valid_times=[(
                        float(row['snippet_start_s']),
                        float(row['snippet_end_s'])
                    )],
                    segment_size_s=window_ms / 1000,
                    length_s=float(row['snippet_end_s']))
                for i, (t1, t2) in enumerate(segments):
                    save_file = os.path.join(save_dir, f"{filename}_{i}.wav")
                    ix1, ix2 = int(t1 * sr), int(t2 * sr)
                    encoded = tf.audio.encode_wav(audio[ix1:ix2], sample_rate=sample_rate)
                    print(save_file)
                    tf.io.write_file(save_file, encoded)


if __name__ == '__main__':
    main()
