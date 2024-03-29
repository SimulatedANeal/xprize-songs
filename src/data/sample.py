from functools import lru_cache

import tensorflow as tf

from src.data.myio import get_wav_file


class AudioSample:

    def __init__(self, filepath, time_start_s, time_end_s, label=None):
        self._file = filepath
        self._start_s = time_start_s
        self._end_s = time_end_s
        self._label = label

    def __str__(self):
        return f"AudioSample(file={self._file}, start={self._start_s}, end={self._end_s}, label={self._label})"

    def __repr__(self):
        return self.__str__()

    def get_data(self, sample_rate):
        audio, _, _ = self.load_audio(filepath=self._file, sample_rate=sample_rate)
        ix1 = int(self._start_s * sample_rate)
        ix2 = int(self._end_s * sample_rate)
        data = tf.squeeze(audio[ix1:ix2], axis=-1)
        return data

    def set_label(self, label, probability):
        self._label = (label, probability)

    @property
    def filepath(self):
        return self._file

    @property
    def start_time(self):
        return self._start_s

    @property
    def end_time(self):
        return self._end_s

    @classmethod
    def join(cls, *samples):
        assert all(s.filepath == samples[0].filepath for s in samples)
        start_time = min(s.start_time for s in samples)
        end_time = max(s.end_time for s in samples)
        return cls(
            filepath=samples[0].filepath,
            time_start_s=start_time,
            time_end_s=end_time)

    @classmethod
    @lru_cache(maxsize=256)
    def load_audio(cls, filepath, sample_rate):
        audio, sr, length_s = get_wav_file(filepath, sample_rate)
        return audio, sr, length_s
