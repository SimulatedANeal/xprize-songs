import argparse
import gc
import os
from collections import defaultdict
from glob import glob

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from data.crickets.ensifera_mingkai.split import test_files
from label import Label
from myio import load_labels, read_wav_tf, read_wav_librosa
from plotting import waveform_and_spectrogram
from transform import get_spectrogram_fn
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


def get_sounds_labels(labels_dir, species):
    all_sound = defaultdict(list)
    file_labels = defaultdict(list)
    labels, _ = load_labels(root_dir=labels_dir, species=species)
    # Goes from smallest to biggest, not reusing anything
    for calltype in Label.non_noise():
        for (t1, t2), filename in labels[calltype]:
            filename = filename.rstrip('.txt')
            all_sound[filename].append((t1, t2))
            if calltype != Label.OTHER_SPECIES.name:
                if not any(overlapping(t1, t2, t3, t4) for (t3, t4) in file_labels[filename]):
                    file_labels[filename].append((t1, t2))
    return all_sound, file_labels


def get_wav_file(filepath, resample_rate=None):
    # print(f"File: {filename}, split: {split}")
    audio_files = []
    for ext in ('wav', 'WAV'):
        audio_files.extend(glob(f"{filepath}.{ext}"))
    audio_file = audio_files[0]  # Should be only one
    if resample_rate:
        audio, sample_rate, length_s = read_wav_librosa(audio_file, resample_rate)
    else:
        audio, sample_rate, length_s = read_wav_tf(audio_file)
    # try:
    #     audio, sample_rate, length_s = read_wav(audio_file)
    # except Exception:
    #     print(audio_file)
    # else:
    #     print(sample_rate, length_s, audio)
    return audio, sample_rate, length_s


def find_ambient_noise(sounds, segment_size_s, end_s):
    # Get segments of ambient noise (no wildlife sounds)
    t2 = 0
    for next_t1, next_t2 in sorted(sounds):
        if next_t1 < t2:
            continue
        last_t2 = t2
        t1, t2 = next_t1, next_t2
        if (t1 - last_t2) > segment_size_s:
            # Window by segment_size / 2
            for t in np.arange(last_t2, t1 - segment_size_s, segment_size_s):
                yield (t, t + segment_size_s)
    if end_s - t2 > segment_size_s:
        for t in np.arange(t2, end_s - segment_size_s, segment_size_s):
            yield (t, t + segment_size_s)


def pad_valid_segments(valid_times, segment_size_s, length_s):
    # Pad or split to correct segment length
    for t1, t2 in valid_times:
        diff = t2 - t1
        avail_left = t1
        avail_right = length_s - t2
        to_pad = segment_size_s - diff
        if to_pad > 0.:
            new_t1, new_t2 = pad(t1, t2, to_pad, avail_left, avail_right)
            yield (new_t1, new_t2)
        elif to_pad < 0.:
            split_point = t1 + (np.random.random() * segment_size_s)
            while split_point < t2:
                new_t1, new_t2 = pad(
                    t1, split_point,
                    pad_by=segment_size_s - (split_point - t1),
                    available_left=avail_left,
                    available_right=length_s - split_point)
                yield (new_t1, new_t2)
                split_point = new_t2 + (np.random.random() * segment_size_s)
                t1 = new_t2
                avail_left = t1
            # Add last segment
            final_t1, final_t2 = pad(
                new_t2, t2,
                pad_by=segment_size_s - (t2 - new_t2),
                available_left=avail_left,
                available_right=avail_right)
            yield (final_t1, final_t2)
        else:
            yield (t1, t2)



def main():
    args = get_args()
    labels_dir = args.directory_labels
    audio_dir = args.directory_audio
    save_dir = args.directory_save
    species_list = get_species_list(root_dir=labels_dir)

    segment_size_ms = args.window_size_ms
    segment_size_s = segment_size_ms / MS

    save_dir = os.path.join(save_dir, f'{segment_size_ms}ms')
    if args.resample_to_hz:
        save_dir = f"{save_dir}_{int(args.resample_to_hz / 1000)}khz"

    # spectro_fn = get_spectrogram_fn(
    #     sample_rate=args.resample_to_hz,
    #     nfft=1024,
    #     window_ms=2,
    #     stride_ms=1)

    for species in tqdm(species_list, total=len(species_list)):
        print(species)
        all_sound, file_labels = get_sounds_labels(labels_dir, species)
        for filename, times in tqdm(file_labels.items(), total=len(file_labels)):
            split = 'test' if filename in test_files[species] else 'train'
            filepath = os.path.join(audio_dir, species, filename)
            audio, sample_rate, length_s = get_wav_file(
                filepath, resample_rate=args.resample_to_hz)
            # example_segment = audio[:int(segment_size_s * sample_rate.numpy())]
            # print(example_segment)
            # waveform_and_spectrogram(example_segment, spectro_fn, show_shape=True)
            sr = sample_rate.numpy()

            save_gen = pad_valid_segments(
                valid_times=times,
                segment_size_s=segment_size_s,
                length_s=length_s)
            save_loc = os.path.join(save_dir, split, species)
            for i, (t1, t2) in enumerate(save_gen):
                save_file = os.path.join(save_loc, f"{filename}_{i}.wav")
                ix1, ix2 = int(t1 * sr), int(t2 * sr)
                encoded = tf.audio.encode_wav(audio[ix1:ix2], sample_rate=sample_rate)
                tf.io.write_file(save_file, encoded)

            save_gen = find_ambient_noise(
                sounds=all_sound[filename],
                segment_size_s=segment_size_s,
                end_s=length_s)
            save_loc = os.path.join(save_dir, split, 'ambient')
            for i, (t1, t2) in enumerate(save_gen):
                save_file = os.path.join(save_loc, f"{filename}_{i}.wav")
                ix1, ix2 = int(t1 * sr), int(t2 * sr)
                encoded = tf.audio.encode_wav(audio[ix1:ix2], sample_rate=sample_rate)
                tf.io.write_file(save_file, encoded)

        del audio, save_gen, all_sound, file_labels
        tf.keras.backend.clear_session()
        gc.collect()


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
        "--window-size-ms", "-w", type=int, default=250,
        help="Audio segment window size in ms")
    parser.add_argument(
        "--resample-to-hz", "-r", type=int, default=None,
        help="Resample audio to the specified frequency (hz)."
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()