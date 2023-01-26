import argparse
import io
import pathlib
import struct
import typing as T

import glog
import numpy as np
import tensorflow as tf

from waymo_open_dataset.protos.scenario_pb2 import Scenario, Track
from trajectory_util import generate_track_array_segments, compute_shape_descriptor
import descriptors


def make_divisible(num, by):
    """Get `v < num` such that `v` is an exact multiple of `by`."""
    assert num > 0
    assert by > 0
    return num // by * by


def write_shape_descriptors_to(scenario_msg: Scenario, outfile: T.BinaryIO) -> None:
    """Compute shape descriptors for track segments of each track in
    `scenario_msg.tracks` and write them out to `outfile`.
    """
    scenario_id = int(scenario_msg.scenario_id, 16)
    assert scenario_id < 2**64, f"{scenario_id} is too large!"

    for track in scenario_msg.tracks:
        xyv = np.array([(s.center_x, s.center_y, s.valid) for s in track.states])
        AGG_SIZE = 5
        xyv = xyv[:make_divisible(len(xyv), AGG_SIZE)]
        xyv = xyv.reshape((-1, 5, 3)).mean(axis=1)
        SHAPE_DESCR_NPOINTS = 6

        # Iterate over overlapping track segments and produce descriptors.
        for i, seg in enumerate(generate_track_array_segments(xyv)):
            if (seg[:, 2] < 1).any():
                continue  # segments had some invalid state before aggregation.
            seg_descr = descriptors.TrackSegmentShapeDescriptor(
                compute_shape_descriptor(seg[:, :2]), scenario_id, track.id, i)
            outfile.write(seg_descr.raw_bytes())

    assert outfile.tell() % 64 == 0, \
        f"num bytes written out: {outfile.tell()} must be a multiple of 64 for best performance."


def write_track_trajectories_to_files(scenario_msg: Scenario, outdir: pathlib.Path) -> None:
    """Write track trajectories in `scenario_msg` into a single numpy
    zip file under `outdir`. The name of the output file is:
        `f"scen:{scenario_msg.scenario_id}.npz"`
    """
    np.set_printoptions(precision=4, suppress=True)
    tstamps = scenario_msg.timestamps_seconds

    arrays = {}
    for track in scenario_msg.tracks:
        txy = np.array([(t, s.center_x, s.center_y) if s.valid else (t, float('NaN'), float('NaN'))
                        for t, s in zip(tstamps, track.states)],
                       dtype=np.float32)
        arrays[f"{track.id:06d}--{track.object_type:1d}"] = txy

    outfile_path = outdir / f"scen:{scenario_msg.scenario_id}.npz"
    outfile_path.parent.mkdir(parents=True, exist_ok=True)
    with outfile_path.open("wb") as f:
        np.savez_compressed(f, **arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir",
                        "-o",
                        type=pathlib.Path,
                        default=pathlib.Path("."),
                        help="Output directory.")
    parser.add_argument("scenario_tfrecord", type=pathlib.Path)
    args = parser.parse_args()

    trajectory_out_dir = args.outdir / "trajectories"

    descr_out_path = (args.outdir / "descriptors" / f"{args.scenario_tfrecord.name}.descrs")
    descr_out_path.parent.mkdir(parents=True, exist_ok=True)

    with descr_out_path.open("wb") as f:
        dataset = tf.data.TFRecordDataset(args.scenario_tfrecord, compression_type='')
        for str_rec in dataset:
            scenario_msg = Scenario()
            scenario_msg.ParseFromString(str_rec.numpy())
            write_shape_descriptors_to(scenario_msg, f)
            write_track_trajectories_to_files(scenario_msg, trajectory_out_dir)


if __name__ == '__main__':
    main()
