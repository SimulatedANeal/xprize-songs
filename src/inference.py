import argparse
import copy
import csv
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import tensorflow as tf

from src.data.sample import AudioSample
from src.data.myio import list_audio_files
from src.data.transform import window_audio_file
from src.model import load_model


THRESHOLD = 0.5


def get_non_overlapping_segments(sub_segments):
    non_overlapping_segments = []
    for segment in sub_segments:
        is_overlapping = any(
            segment[:2] != other_segment[:2] and is_inside(segment, other_segment)
            for other_segment in sub_segments)
        if not is_overlapping:
            non_overlapping_segments.append(segment)
    return non_overlapping_segments


def is_inside(segment1, segment2):
    return segment2[0] <= segment1[0] and segment1[1] <= segment2[1]


def merge_predictions(call_probability, species_probability, species_list, call_threshold=0.5, species_threshold=0.25):
    n_windows = len(call_probability)

    # Step 1: Identify potential segments based on the "call" classifier
    segments = []
    current_segment = None
    for i in range(n_windows):
        if call_probability[i] >= call_threshold:
            if current_segment is None:
                current_segment = [i, i]
            else:
                current_segment[1] = i
        elif current_segment is not None:
            segments.append(current_segment)
            current_segment = None
    if current_segment is not None:
        segments.append(current_segment)

    # Step 2: Refine segments based on the "species" classifier
    refined_segments = []
    candidates = defaultdict(list)
    for start, end in segments:
        for i in range(start, end + 1):
            for j in range(i, end + 1):
                segment_probs = species_probability[i:j + 1]
                class_probs = np.mean(segment_probs, axis=0)
                class_ixs = np.where(class_probs >= species_threshold)[0]
                if len(class_ixs):
                    for class_ix in class_ixs:
                        candidates[class_ix].append((i, j, class_probs[class_ix]))
                else:
                    class_ix = np.argmax(class_probs)
                    candidates[class_ix].append((i, j, class_probs[class_ix]))
    for class_ix, segments in candidates.items():
        maximal = get_non_overlapping_segments(segments)
        for start, end, probability in maximal:
            refined_segments.append((start, end, species_list[class_ix], probability))
    return refined_segments


def main():
    args = get_args()
    model, config = load_model(args.model_load)
    model_name = "CNN-CricketModel"  # TODO: un-hardcode
    labels = config["labels"]

    new_config = copy.copy(config)
    new_config.pop('labels')
    new_config['model'] = model_name
    sr = config['sample_rate']

    output_dir = os.path.join(args.directory_output, args.directory_data.split('/')[-1])
    output_file = os.path.join(output_dir, 'predictions.csv')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(new_config, f)

    if os.path.exists(output_file):
        print(f"WARNING: Would overwrite existing prediction file {output_file}!")
        user_input = input("Is this okay? (yes/no): ")
        if user_input.strip().lower() == 'yes':
            ...
        else:
            exit(1)

    audio_files = list_audio_files(args.directory_data)
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 't_start', 't_end', 'species', 'probability', 'prediction_timestamp'])
        for filepath in audio_files:
            _, _, length_s = AudioSample.load_audio(filepath, sample_rate=sr)
            windows = window_audio_file(
                parent_filepath=filepath,
                audio_length_s=length_s,
                window_size_ms=config['window_len_ms'],
                stride_ms=config['window_len_ms'])
            batches_call = []
            batches_species = []
            for batch in range(0, len(windows), 1024):
                spectrogram_batch = tf.ragged.stack([
                    model.audio_to_spectrogram(w.get_data(sample_rate=sr))
                    for w in windows[batch:min(batch + 1024, len(windows))]])
                embeddings = model.embed(spectrogram_batch.to_tensor())
                batches_call.append(model.get_call_probability(embeddings).numpy())
                batches_species.append(model.predict_species(embeddings).numpy())
            call_pred = np.concatenate(batches_call, axis=0)
            species_pred = np.concatenate(batches_species, axis=0)
            segments = merge_predictions(
                call_probability=call_pred,
                species_probability=species_pred,
                species_list=labels)
            timestamp = datetime.utcnow()

            for segment in segments:
                new_sample = AudioSample.join(*windows[segment[0]:segment[1] + 1])
                writer.writerow([
                    new_sample.filepath,
                    new_sample.start_time,
                    new_sample.end_time,
                    segment[2],  # species prediction
                    segment[3],  # prediction probability
                    timestamp,
                ])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-load", "-m", type=str, default=None)
    parser.add_argument("--directory-data", "-d", type=str, required=True)
    parser.add_argument("--directory-output", "-o", type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()