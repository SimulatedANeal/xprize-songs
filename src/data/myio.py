import os
from collections import defaultdict
from glob import glob

import librosa
import tensorflow as tf

from src.data.label import Label


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
            label_dict[label.name].append(((float(start), float(stop)), filename))
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


def list_audio_files(directory, filename=None, extension=('wav', 'WAV')):
    audio_files = []
    path = os.path.join(directory, filename if filename else '*')
    for ext in extension:
        audio_files.extend(glob(f"{path}.{ext}"))
    return audio_files


def get_wav_file(filepath, resample_rate=None):
    if resample_rate:
        audio, sample_rate, length_s = read_wav_librosa(filepath, resample_rate)
    else:
        audio, sample_rate, length_s = read_wav_tf(filepath)
    # try:
    #     audio, sample_rate, length_s = read_wav(audio_file)
    # except Exception:
    #     print(audio_file)
    # else:
    #     print(sample_rate, length_s, audio)
    return audio, sample_rate, length_s


def read_wav_tf(filepath):
    file_contents = tf.io.read_file(filepath)
    audio, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int32)
    length_s = audio.shape[0] / sample_rate.numpy()
    return audio, sample_rate, length_s


def read_wav_librosa(filepath, resample_rate):
    y, s = librosa.load(filepath, sr=resample_rate)
    audio = tf.constant(y, dtype=tf.float32)
    audio = tf.expand_dims(audio, axis=-1)
    sample_rate = tf.constant(s, dtype=tf.int32)
    length_s = audio.shape[0] / sample_rate.numpy()
    return audio, sample_rate, length_s
