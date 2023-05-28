import argparse
import csv
from collections import defaultdict

import numpy as np
import tensorflow as tf

from src.data.sample import AudioSample
from src.data.myio import list_audio_files, get_wav_file
from src.data.plotting import waveform_and_spectrogram
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


def merge_predictions(call_probability, species_probability, species_list, call_threshold=0.75):
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
    labels = config["labels"]
    audio_files = list_audio_files(args.directory_data)
    with open(args.output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 't_start', 't_end', 'species', 'probability'])
        for filepath in audio_files:
            audio, sample_rate, length_s = get_wav_file(
                filepath,
                resample_rate=config['sample_rate'])
            windows = window_audio_file(
                waveform=audio,
                sample_rate=sample_rate,
                parent_filepath=filepath,
                window_size_ms=config['window_len_ms'],
                stride_ms=config['window_len_ms'])
            spectrogram_batch = tf.stack([model.audio_to_spectrogram(w.data) for w in windows])
            embeddings = model.embed(spectrogram_batch)
            call_pred = model.get_call_probability(embeddings).numpy()
            species_pred = model.predict_species(embeddings).numpy()

            segments = merge_predictions(
                call_probability=call_pred,
                species_probability=species_pred,
                species_list=labels)

            for segment in segments:
                new_sample = AudioSample.join(*windows[segment[0]:segment[1] + 1])
                # waveform_and_spectrogram(
                #     waveform=new_sample.data,
                #     spectrogram_fn=model.audio_to_spectrogram,
                #     label=f"{segment[2]}: {segment[3] * 100:.2f}%"
                # )
                writer.writerow([
                    new_sample.filepath,
                    new_sample.start_time,
                    new_sample.end_time,
                    segment[2],  # species prediction
                    segment[3],  # prediction probability
                ])



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-load", "-m", type=str, default=None)
    parser.add_argument("--directory-data", "-d", type=str, required=True)
    parser.add_argument("--output-file", "-o", type=str, default='predictions.txt')
    return parser.parse_args()


if __name__ == '__main__':
    main()