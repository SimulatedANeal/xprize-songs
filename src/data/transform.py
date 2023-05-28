import tensorflow as tf

from src.data.sample import AudioSample

MS = 1000


def get_spectrogram_fn(sample_rate, nfft, window_ms=2, stride_ms=0.5, max_freq_hz=None):
    window = int(sample_rate * window_ms / MS)
    stride = int(sample_rate * stride_ms / MS)
    print(f"Using nfft {nfft}, window {window}, and stride {stride}")

    def get_spectrogram(waveform):
        # Convert the waveform to a spectrogram via STFT.
        spectrogram = tf.signal.stft(waveform, fft_length=nfft, frame_length=window, frame_step=stride)
        # Obtain the magnitude of the STFT.
        spectrogram = tf.abs(spectrogram)
        # Optionally truncate frequency
        if max_freq_hz:
            max_y = int(max_freq_hz / sample_rate * nfft)
            spectrogram = spectrogram[:, :max_y]
        # Add a `channels` dimension
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram

    return get_spectrogram


def window(waveform, sample_rate, window_size_ms=500, stride_ms=100):
    samples_per_window = int(sample_rate * window_size_ms / MS)
    stride = int(sample_rate * stride_ms / MS)
    print(f"{samples_per_window} samples per window with stride {stride}")
    windows = [
        waveform[ix:ix+samples_per_window]
        for ix in range(0, len(waveform) - samples_per_window, stride)]
    return windows


def window_audio_file(waveform, sample_rate, parent_filepath, window_size_ms=500, stride_ms=100):
    samples_per_window = int(sample_rate * window_size_ms / MS)
    stride = int(sample_rate * stride_ms / MS)
    windows = [
        AudioSample(
            filepath=parent_filepath,
            raw_audio_data=waveform[ix:ix + samples_per_window],
            time_start_s=(ix / sample_rate).numpy(),
            time_end_s=((ix + samples_per_window) / sample_rate).numpy())
        for ix in range(0, len(waveform) - samples_per_window, stride)]
    if len(waveform) % samples_per_window:  # not evenly divisible
        # Add last window to get final bit of audio
        windows.append(
            AudioSample(
                filepath=parent_filepath,
                raw_audio_data=waveform[len(waveform) - samples_per_window:],
                time_start_s=((len(waveform) - samples_per_window) / sample_rate).numpy(),
                time_end_s=(len(waveform) / sample_rate).numpy()))
    return windows
