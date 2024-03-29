"""
"""

import numpy as np

__all__ = [
    "schmidt_spike_removal",
]


def schmidt_spike_removal(
    original_signal: np.ndarray,
    fs: int,
    window_size: float = 0.5,
    threshold: float = 3.0,
    eps: float = 1e-4,
) -> np.ndarray:
    """Spike removal using Schmidt algorithm [1]_.

    Parameters
    ----------
    original_signal : numpy.ndarray
        The original signal.
    fs : int
        Sampling frequency of the signal.
    window_size : float, default 0.5
        Sliding window size, with units in seconds.
    threshold : float, default 3.0
        Threshold (multiplier for the median value) for detecting spikes.
    eps : float, default 1e-4
        Epsilon for numerical stability.

    Returns
    -------
    despiked_signal : numpy.ndarray
        Despiked signal.

    References
    ----------
    .. [1] Schmidt, Samuel E., et al. "Segmentation of heart sound recordings by a duration-dependent hidden Markov model."
           Physiological measurement 31.4 (2010): 513.

    """
    window_size = round(fs * window_size)
    nframes, res = divmod(original_signal.shape[0], window_size)
    frames = original_signal[: window_size * nframes].reshape((nframes, window_size))
    if res > 0:
        nframes += 1
        frames = np.concatenate((frames, original_signal[-window_size:].reshape((1, window_size))), axis=0)
    MAAs = np.abs(frames).max(axis=1)  # of shape (nframes,)

    while len(np.where(MAAs > threshold * np.median(MAAs))[0]) > 0:
        frame_num = np.where(MAAs == MAAs.max())[0][0]
        spike_position = np.argmax(np.abs(frames[frame_num]))
        zero_crossings = np.where(np.diff(np.sign(frames[frame_num])))[0]
        spike_start = np.where(zero_crossings <= spike_position)[0]
        spike_start = zero_crossings[spike_start[-1]] if len(spike_start) > 0 else 0
        spike_end = np.where(zero_crossings >= spike_position)[0]
        spike_end = zero_crossings[spike_end[0]] + 1 if len(spike_end) > 0 else window_size
        frames[frame_num, spike_start:spike_end] = eps
        MAAs = np.abs(frames).max(axis=1)

    despiked_signal = original_signal.copy()
    if res > 0:
        despiked_signal[-window_size:] = frames[-1]
        nframes -= 1
    despiked_signal[: window_size * nframes] = frames[:nframes, ...].reshape((-1,))

    return despiked_signal
