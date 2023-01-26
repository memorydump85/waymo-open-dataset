import numpy as np
import scipy


def generate_track_array_segments(xyv: np.ndarray, seg_len: int = 6) -> np.ndarray:
    for i in range(0, len(xyv) - seg_len):
        yield xyv[i:i + seg_len]


def compute_shape_descriptor(seg_xy: np.ndarray) -> np.ndarray:
    """Compute a vector description of the shape of `seg_xy` based on
    pairwise xy-point distances.
    """
    pair_dists = scipy.spatial.distance_matrix(seg_xy, seg_xy).astype(np.float32)
    descr = np.hstack([np.diag(pair_dists, k) for k in (1, 2, 3)])
    assert len(descr.shape) == 1
    assert descr.dtype == np.float32
    return descr
