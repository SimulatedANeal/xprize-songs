import numpy as np
import tensorflow as tf


class AudioSample:

    def __init__(self, filepath, raw_audio_data, time_start_s, time_end_s, label=None):
        self._file = filepath
        self._data = raw_audio_data
        self._start_s = time_start_s
        self._end_s = time_end_s
        self._label = label

    def __str__(self):
        return f"AudioSample(file={self._file}, start={self._start_s}, end={self._end_s}, label={self._label})"

    def __repr__(self):
        return self.__str__()

    def set_label(self, label, probability):
        self._label = (label, probability)

    @property
    def filepath(self):
        return self._file

    @property
    def data(self):
        data = self._data
        if len(data.shape) > 1:
            data = tf.squeeze(data, axis=-1)
        return data

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
            raw_audio_data=np.concatenate([s.data for s in samples]),
            time_start_s=start_time,
            time_end_s=end_time)

