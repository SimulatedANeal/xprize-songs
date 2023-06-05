import argparse
import copy
import csv
import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

from src.data.sample import AudioSample
from src.data.myio import list_audio_files
from src.data.transform import window_audio_file
from src.model import load_model


def get_call_snippets(call_probability, call_threshold=0.5):
    n_windows = len(call_probability)
    current_segment = None
    for i in range(n_windows):
        if call_probability[i] >= call_threshold:
            if current_segment is None:
                current_segment = [i, i]
            elif (i - current_segment[0]) > 10:
                # Cap size of prediction segment
                yield current_segment
                current_segment = [i, i]
            else:
                current_segment[1] = i
        elif current_segment is not None:
            yield current_segment
            current_segment = None
    if current_segment is not None:
        yield current_segment


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
    field_names = [
        'source', 'snippet_t_start_s', 'snippet_t_end_s',
        'call_probability', 'top_species', 'top_species_probability', 'prediction_timestamp']
    field_names += labels
    with open(output_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for filepath in audio_files:
            try:
                _, _, length_s = AudioSample.load_audio(filepath, sample_rate=sr)
            except EOFError:
                print(f"WARNING! Skipping corrupted file: {filepath}")
                continue
            print(f"Predicting on {filepath}")
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

            timestamp = datetime.utcnow()
            call_pred = np.concatenate(batches_call, axis=0)
            species_pred = np.concatenate(batches_species, axis=0)

            for i, j in get_call_snippets(call_probability=call_pred):
                new_sample = AudioSample.join(*windows[i:j + 1])
                segment = species_pred[i:j + 1]
                call_prob = np.mean(call_pred[i:j + 1])
                class_probs = np.mean(segment, axis=0)
                predicted_class = np.argmax(class_probs)
                rowdict = {
                    'source': new_sample.filepath,
                    'snippet_t_start_s': new_sample.start_time,
                    'snippet_t_end_s': new_sample.end_time,
                    'detection_method': model_name,
                    'call_probability': call_prob,
                    'top_species': labels[predicted_class],
                    'top_species_probability': class_probs[predicted_class],
                    'prediction_timestamp': timestamp}
                for class_ix, probability in enumerate(class_probs):
                    rowdict[labels[class_ix]] = probability
                writer.writerow(rowdict)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-load", "-m", type=str, default=None)
    parser.add_argument("--directory-data", "-d", type=str, required=True)
    parser.add_argument("--directory-output", "-o", type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()