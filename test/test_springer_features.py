"""
"""

from pathlib import Path

from pytest import approx
import numpy as np
import scipy.io.wavfile as siw

try:
    from pcg_springer_features import get_springer_features, get_schmidt_heart_rate
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1].resolve()))

    from pcg_springer_features import get_springer_features, get_schmidt_heart_rate


_PROJECT_DIR = Path(__file__).parents[1].resolve()
_SAMPLE_DATA_DIR = _PROJECT_DIR / "sample_data"


def _to_dtype(data: np.ndarray, dtype: np.dtype = np.float32) -> np.ndarray:
    """ """
    if data.dtype == dtype:
        return data
    if data.dtype in (np.int8, np.uint8, np.int16, np.int32, np.int64):
        data = data.astype(dtype) / (np.iinfo(data.dtype).max + 1)
    return data


def test_extract_features():
    """ """
    fs, data = siw.read(_SAMPLE_DATA_DIR / "13918_AV.wav")
    data = _to_dtype(data, np.float32)

    mean_hr, systolic_time_interval = get_schmidt_heart_rate(data, fs)
    assert mean_hr == approx(105, abs=0.1)

    feature_fs = 50
    springer_features = get_springer_features(data, fs, feature_fs)
    assert springer_features.shape[0] == 4 * int(
        np.ceil(data.shape[0] / (fs / feature_fs))
    )


if __name__ == "__main__":
    test_extract_features()
