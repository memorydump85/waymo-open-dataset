import argparse
import io
import pathlib
import struct
import typing as T

import glog
import numpy as np
import scipy
import tensorflow as tf

from waymo_open_dataset.protos.scenario_pb2 import Scenario, Track


def make_divisible(num, by):
    """Decrease `num` so that it is exactly divisible by `by`."""
    assert num > 0
    assert by > 0
    return num // by * by


def write_shape_descriptors_to(scenario_msg: Scenario, outfile: T.BinaryIO) -> None:
    scenario_id = int(scenario_msg.scenario_id, 16)
    assert scenario_id < 2**64, f"{scenario_id} is too large!"

    for track in scenario_msg.tracks:
        xyv = np.array([(s.center_x, s.center_y, s.valid) for s in track.states])
        AGG_SIZE = 5
        xyv = xyv[:make_divisible(len(xyv), AGG_SIZE)]
        xyv = xyv.reshape((-1, 5, 3)).mean(axis=1)
        SHAPE_DESCR_NPOINTS = 6
        
        # Iterate over overlapping track segments and produce descriptors.
        for i in range(0, len(xyv) - SHAPE_DESCR_NPOINTS):
            seg = xyv[i: i + SHAPE_DESCR_NPOINTS]
            if (seg[:, 2] < 1).any():
                continue  # segments had some invalid state before aggregation.
            seg_xy = seg[:, :2]
            pair_dists = scipy.spatial.distance_matrix(seg_xy, seg_xy).astype(np.float32)
            descr = np.hstack([np.diag(pair_dists, k=1),
                               np.diag(pair_dists, k=2),
                               np.diag(pair_dists, k=3)])
            assert len(descr.shape) == 1
            assert descr.dtype == np.float32
            outfile.write(descr.tobytes(order='C'))
            outfile.write(struct.pack('=QLL', scenario_id, track.id, i))

    assert outfile.tell() % 64 == 0, \
        f"num bytes written out: {outfile.tell()} must be a multiple of 64"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", "-o", type=pathlib.Path,
                        default=pathlib.Path("."),
                        help="Output directory.")
    parser.add_argument("scenario_tfrecord", type=pathlib.Path)
    args = parser.parse_args()

    outfile_path = args.outdir / "descriptors" / f"{args.scenario_tfrecord.name}.descrs"
    outfile_path.parent.mkdir(parents=True, exist_ok=True)
    with outfile_path.open("wb") as f:
        dataset = tf.data.TFRecordDataset(args.scenario_tfrecord, compression_type='')
        buf = io.BytesIO()
        for str_rec in dataset:
            scenario_msg = Scenario()
            scenario_msg.ParseFromString(str_rec.numpy())
            write_shape_descriptors_to(scenario_msg, f)

if __name__ == '__main__':
    main()
