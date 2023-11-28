"""
"""

from typing import Optional, Union

import numpy as np
import pywt
from easydict import EasyDict as ED

__all__ = [
    "get_dwt_features",
    "get_full_dwt_features",
]


def get_dwt_features(signal: np.ndarray, fs: int, config: Optional[dict] = None) -> np.ndarray:
    """Compute the discrete wavelet transform (DWT)
    features using Springer's algorithm.

    For Springer's algorithm, see [1]_.

    Parameters
    ----------
    signal : numpy.ndarray
        The (PCG) signal, of shape ``(nsamples,)``.
    fs : int
        Sampling frequency of the signal.
    config : dict, optional
        The configuration for computing the features, with the following items:

        - ``'wavelet_level'`` : int,
          the level of the wavelet decomposition, default: 3
        - ``'wavelet_name'`` : str,
          the name of the wavelet, default: "db7"

    Returns
    -------
    dwt_features : numpy.ndarray
        The DWT features, of shape ``(nsamples,)``.

    References
    ----------
    .. [1] Springer, David B., Lionel Tarassenko, and Gari D. Clifford. "Logistic regression-HSMM-based heart sound segmentation."
           IEEE transactions on biomedical engineering 63.4 (2015): 822-832.

    """
    cfg = ED(
        wavelet_level=3,
        wavelet_name="db7",
    )
    cfg.update(config or {})
    siglen = len(signal)

    detail_coefs = pywt.downcoef("d", signal, wavelet=cfg.wavelet_name, level=cfg.wavelet_level)
    dwt_features = _wkeep1(np.repeat(detail_coefs, 2**cfg.wavelet_level), siglen)
    return dwt_features


def get_full_dwt_features(signal: np.ndarray, fs: int, config: Optional[dict] = None) -> np.ndarray:
    """Compute the full DWT features using Springer's algorithm.

    Parameters
    ----------
    signal : numpy.ndarray
        The (PCG) signal, of shape ``(nsamples,)``.
    fs : int
        The sampling frequency.
    config : dict, optional
        The configuration, with the following keys:

        - ``'wavelet_level'``: int,
          the level of the wavelet decomposition, default: 3
        - ``'wavelet_name'``: str,
          the name of the wavelet, default: "db7"

    Returns
    -------
    dwt_features : numpy.ndarray
        The full DWT features, of shape ``(wavelet_level, nsamples)``.

    """
    cfg = ED(
        wavelet_level=3,
        wavelet_name="db7",
    )
    cfg.update(config or {})
    siglen = len(signal)

    detail_coefs = pywt.wavedec(signal, cfg.wavelet_name, level=cfg.wavelet_level)[:0:-1]
    dwt_features = np.zeros((cfg.wavelet_level, siglen), dtype=signal.dtype)
    for i, detail_coef in enumerate(detail_coefs):
        dwt_features[i] = _wkeep1(np.repeat(detail_coef, 2 ** (i + 1)), siglen)
    return dwt_features


def _wkeep1(x: np.ndarray, k: int, opt: Union[str, int] = "c") -> np.ndarray:
    """
    modified from the matlab function ``wkeep1``

    Parameters
    ----------
    x : numpy.ndarray
        The input array.
    k : int
        The length of the output array。
    opt : str or int, optional
        Specifies the position of the output array in the input array,
        if ``opt`` is an integer, then it is the first index of the output array,
        if ``opt`` is a string, then it can be one of the following:
        - ``"c"`` or ``"center"`` or ``"centre"``: the output array is centered in the input array
        - ``"l"`` or ``"left"``: the output array is left-aligned in the input array
        - ``"r"`` or ``"right"``: the output array is right-aligned in the input array

    Returns
    -------
    y : numpy.ndarray
        The output array, of shape ``(k,)``,
        if ``k > len(x)``, then ``x`` is returned directly.

    References
    ----------
    `wkeep1.m` of the matlab wavelet toolbox.

    """
    x_len = len(x)
    if x_len <= k:
        return x
    if isinstance(opt, int):
        first = opt
    elif opt.lower() in ["c", "center", "centre"]:
        first = (x_len - k) // 2
    elif opt.lower() in ["l", "left"]:
        first = 0
    elif opt.lower() in ["r", "right"]:
        first = x_len - k
    else:
        raise ValueError(f"Unknown option: {opt}")
    assert 0 <= first <= x_len - k, f"Invalid first index: {first}"
    return x[first : first + k]
