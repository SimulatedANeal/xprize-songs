import argparse
import os
import random
from collections import defaultdict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tensorflow_io as tfio
import tensorflow as tf

from data.crickets.ensifera_mingkai.split import test_files
from label import Label
from myio import load_labels
from util import get_species_list


MS = 1000.
INTSIZE = 32768.0


def overlapping(t1, t2, start, stop):
    return (t1 <= start and t2 >= start) or (t1 <= stop and t2 >= stop)


def pad(t1, t2, pad_by, available_left, available_right):
    at_least_left = 0.
    if available_right < pad_by:
        at_least_left = (1. - (available_right / pad_by)) * pad_by
        available_left -= at_least_left
    pad_left = at_least_left + np.random.random() * min(
        (pad_by - at_least_left) * 7 / 8,
        available_left)
    pad_right = pad_by - pad_left
    new_t1 = t1 - pad_left
    new_t2 = t2 + pad_right
    return new_t1, new_t2


def read_wav(filepath):
    file_contents = tf.io.read_file(filepath)
    audio, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # audio = tf.squeeze(audio, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int32)
    length_s = audio.shape[0] / sample_rate.numpy()
    return audio, sample_rate, length_s


def save_wav_slice(wav, sample_rate, start_s, stop_s, filename):
    rate = sample_rate.numpy()
    ix1, ix2 = int(start_s * rate), int(stop_s * rate)
    sample = wav[ix1:ix2]
    encoded = tf.audio.encode_wav(sample, sample_rate=sample_rate)
    tf.io.write_file(filename, encoded)

def main():
    args = get_args()
    labels_dir = args.directory_labels
    audio_dir = args.directory_audio
    save_dir = args.directory_save
    species_list = get_species_list(root_dir=labels_dir)

    segment_size_ms = args.window_size_ms
    segment_size_s = segment_size_ms / MS

    for species in species_list:
        print(species)
        labels, _ = load_labels(root_dir=labels_dir, species=species)
        all_sound = defaultdict(list)
        file_labels = defaultdict(list)
        # Goes from smallest to biggest, not reusing anything
        for calltype in Label.non_noise():
            for (t1, t2), filename in labels[calltype]:
                filename = filename.rstrip('.txt')
                all_sound[filename].append((t1, t2))
                if not any(overlapping(t1, t2, t3, t4) for (t3, t4) in file_labels[filename]):
                    file_labels[filename].append((t1, t2))

        for filename, times in file_labels.items():
            split = 'test' if filename in test_files[species] else 'train'
            print(f"File: {filename}, split: {split}")
            audio_files = []
            for ext in ('wav', 'WAV'):
                audio_files.extend(glob(os.path.join(audio_dir, species, f'{filename}.{ext}')))
            audio_file = audio_files[0]  # Should be only one
            audio, sample_rate, length_s = read_wav(audio_file)
            print(sample_rate, length_s, audio)

            # Pad or split to correct segment length
            segments_to_save = []
            for t1, t2 in times:
                diff = t2 - t1
                avail_left = t1
                avail_right = length_s - t2
                to_pad = segment_size_s - diff
                if to_pad > 0.:
                    new_t1, new_t2 = pad(t1, t2, to_pad, avail_left, avail_right)
                    segments_to_save.append((new_t1, new_t2))
                elif to_pad < 0.:
                    split_point = t1 + (np.random.random() * segment_size_s)
                    while split_point < t2:
                        new_t1, new_t2 = pad(
                            t1, split_point,
                            pad_by=segment_size_s - (split_point - t1),
                            available_left=avail_left,
                            available_right=length_s - split_point)
                        segments_to_save.append((new_t1, new_t2))
                        split_point = new_t2 + (np.random.random() * segment_size_s)
                        t1 = new_t2
                        avail_left = t1
                    # Add last segment
                    final_t1, final_t2 = pad(
                        new_t2, t2,
                        pad_by=segment_size_s - (t2 - new_t2),
                        available_left=avail_left,
                        available_right=avail_right)
                    segments_to_save.append((final_t1, final_t2))
                else:
                    segments_to_save.append((t1, t2))

            # Get segments of ambient noise (no wildlife sounds)
            silence_to_save = []
            sounds = sorted(all_sound[filename])
            t2 = 0
            for next_t1, next_t2 in sounds:
                if next_t1 < t2:
                    continue
                last_t2 = t2
                t1, t2 = next_t1, next_t2
                if (t1 - last_t2) > segment_size_s:
                    # Window by segment_size / 2
                    for t in np.arange(last_t2, t1, segment_size_s / 2)[:-2]:
                        silence_to_save.append((t, t + segment_size_s))
                    silence_to_save.append((t1 - segment_size_s, t1))
            if length_s - t2 > segment_size_s:
                for t in np.arange(t2, length_s, segment_size_s / 2)[:-2]:
                    silence_to_save.append((t, t + segment_size_s))
                silence_to_save.append((length_s - segment_size_s, length_s))

            silence_to_save = [
                (t1, t2) for t1, t2 in silence_to_save
                if not any(overlapping(t1, t2, t3, t4) for t3, t4 in sounds)]

            # rate = sample_rate.numpy()
            # window = int(rate / MS)  # 1ms windows for spectogram
            # stride = int(window / 2)
            # for t1, t2 in random.sample(silence_to_save, min(2, len(silence_to_save))):
            #     ix1, ix2 = int(t1 * rate), int(t2 * rate)
            #     sample = audio[ix1:ix2]
            #     tensor = tf.cast(sample, tf.float32) / INTSIZE
            #     tensor = tf.squeeze(tensor)
            #     spectrogram = tfio.audio.spectrogram(
            #         tensor, nfft=512, window=window, stride=stride)
            #     plt.figure()
            #     plt.imshow(tf.math.log(spectrogram).numpy().T)
            #     plt.show()

            save_loc = os.path.join(save_dir, split, species)
            for i, (t1, t2) in enumerate(segments_to_save):
                save_file = os.path.join(save_loc, f"{filename}_{i}.wav")
                save_wav_slice(
                    wav=audio, sample_rate=sample_rate,
                    start_s=t1, stop_s=t2,
                    filename=save_file)

            save_loc = os.path.join(save_dir, split, 'ambient')
            for i, (t1, t2) in enumerate(silence_to_save):
                save_file = os.path.join(save_loc, f"{filename}_{i}.wav")
                save_wav_slice(
                    wav=audio, sample_rate=sample_rate,
                    start_s=t1, stop_s=t2,
                    filename=save_file)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory-labels", "-l", type=str, required=True,
        help="Directory where labels (species directories) are located")
    parser.add_argument(
        "--directory-audio", "-a", type=str, required=True,
        help="Directory where audio files (species directories) are located.")
    parser.add_argument(
        "--directory-save", "-s", type=str, required=True,
        help="Directory where segmented audio examples will be saved.")
    parser.add_argument(
        "--window-size-ms", "-w", type=int, default=256,
        help="Audio segment window size in ms")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()