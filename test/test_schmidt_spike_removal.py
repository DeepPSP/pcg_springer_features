""" """

from pathlib import Path

import numpy as np
import scipy.io.wavfile as siw

try:
    from pcg_springer_features.schmidt_spike_removal import schmidt_spike_removal
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1].resolve()))

    from pcg_springer_features.schmidt_spike_removal import schmidt_spike_removal


def _to_dtype(data: np.ndarray, dtype: np.dtype = np.float32) -> np.ndarray:
    """ """
    if data.dtype == dtype:
        return data
    if data.dtype in (np.int8, np.uint8, np.int16, np.int32, np.int64):
        data = data.astype(dtype) / (np.iinfo(data.dtype).max + 1)
    return data


def test_schmidt_spike_removal():
    """ """
    fs, data = siw.read(Path(__file__).parents[1] / "sample_data" / "13918_AV.wav")
    data = _to_dtype(data, np.float32)
    window_size = 0.5

    for spikes_density in [0.3, 0.01, 0.001, 0.0001]:
        original_signal = data.copy()
        # put spikes in the first frame
        spikes_pos = np.random.choice(
            int(fs * window_size),
            int(fs * window_size * spikes_density),
            replace=False,
        )
        original_signal[spikes_pos] = 10 * np.max(np.abs(original_signal))

        despiked_signal = schmidt_spike_removal(original_signal, fs)

        despiked_signal = schmidt_spike_removal(despiked_signal, fs)


if __name__ == "__main__":
    test_schmidt_spike_removal()
    print("test_schmidt_spike_removal passed!")
