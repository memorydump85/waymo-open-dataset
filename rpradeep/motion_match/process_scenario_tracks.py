"""Process scenarios in a motion dataset tfrecord file and index the
tracks within each scenario for motion search.

When we process the tracks within each `Scenario` proto of the input
dataset, we write out the following information under the path passed as
--outdir to this script:

1. We iterate over overlapping track-segments within each track and we
write out a descriptor that describes the shape of the motion for that
segment. See `write_shape_descriptors_to` for details.

2. We write out the (scenario_id, track_id, num_segments) for the track
segment whose descriptor was written out into a separate output byte
stream.

The information written out to the two separate streams can be
associated by index / sequence number. The streams are backed by files
stored under OUTDIR/descriptors.

3. For each scenario, we also write out position information for each
track into a file under OUTDIR/trajectories. This information can be
used for visualizing tracks, given (scenario_id, track_id), without
incurring the overhead of parsing through tfrecords.
"""

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

EXPECTED_TRACK_SEGMENT_DESCR_SIZE = 12


def make_divisible(num, by):
    """Get `v < num` such that `v` is an exact multiple of `by`."""
    assert num > 0
    assert by > 0
    return num // by * by


def generate_track_array_segments(xyv: np.ndarray,
                                  seg_len: int = 6) -> np.ndarray:
    for i in range(0, len(xyv) - seg_len):
        yield xyv[i:i + seg_len]


def compute_shape_descriptor(seg_xy: np.ndarray) -> np.ndarray:
    """Compute a vector description of the shape of `seg_xy` based on
    pairwise xy-point distances.
    """
    pair_dists = scipy.spatial.distance_matrix(seg_xy,
                                               seg_xy).astype(np.float32)
    descr = np.hstack([np.diag(pair_dists, k) for k in (1, 2, 3)])
    assert len(descr.shape) == 1
    assert len(descr) == EXPECTED_TRACK_SEGMENT_DESCR_SIZE
    assert descr.dtype == np.float32
    return descr


def write_shape_descriptors_to(scenario_msg: Scenario,
                               descr_outfile: T.BinaryIO,
                               idinfo_outfile: T.BinaryIO) -> None:
    """Compute shape descriptors for track segments of each track in
    `scenario_msg.tracks` and write them out to `descr_outfile`.

    Write out the following info into `idinfo_outfile`:
        scenario_id: uint64
        track_id: uint32
        num_segments: uint32
        (total: 16 bytes of ID info)

    `descr_outfile` will be mapped into memory for similarity search;
    The result indices returned by the similarity search can be used to
    lookup information serialized into `idinfo_outfile`.
    """
    scenario_id = int(scenario_msg.scenario_id, 16)
    assert scenario_id < 2**64, f"{scenario_id} is too large!"

    for track in scenario_msg.tracks:
        xyv = np.array([(s.center_x, s.center_y, s.valid)
                        for s in track.states])
        AGG_SIZE = 5  # Aggregate .5 seconds worth of motion for smoothness.
        xyv = xyv[:make_divisible(len(xyv), AGG_SIZE)]
        xyv = xyv.reshape((-1, 5, 3)).mean(axis=1)
        SHAPE_DESCR_NPOINTS = 6  # = 3 seconds of motion.

        # Iterate over overlapping track segments and produce descriptors.
        for i, seg in enumerate(generate_track_array_segments(xyv)):
            if (seg[:, 2] < 1).any():
                continue  # segments had some invalid state before aggregation.
            seg_descr = compute_shape_descriptor(seg[:, :2])
            descr_outfile.write(seg_descr.tobytes())

        num_segments = i
        idinfo = struct.pack('@QII', scenario_id, track.id, num_segments)
        idinfo_outfile.write(idinfo)

    assert idinfo_outfile.tell() % 8 == 0, \
        f"num bytes written out: {idinfo_outfile.tell()} must be a multiple of 8 for best performance."
    assert descr_outfile.tell() % 8 == 0, \
        f"num bytes written out: {descr_outfile.tell()} must be a multiple of 8 for best performance."


def write_track_trajectories_to_files(scenario_msg: Scenario,
                                      outdir: pathlib.Path) -> None:
    """Write track trajectories in `scenario_msg` into a single numpy
    zip file under `outdir`. The name of the output file is:
        `f"scen:{scenario_msg.scenario_id}.npz"`
    """
    np.set_printoptions(precision=4, suppress=True)
    tstamps = scenario_msg.timestamps_seconds

    arrays = {}
    for track in scenario_msg.tracks:
        txy = np.array([(t, s.center_x, s.center_y) if s.valid else
                        (t, float('NaN'), float('NaN'))
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

    descr_dir = args.outdir / "descriptors"
    descr_dir.mkdir(parents=True, exist_ok=True)
    descr_out_path = descr_dir / f"{args.scenario_tfrecord.name}.descrs"
    idinfo_out_path = descr_dir / f"{args.scenario_tfrecord.name}.idinfo"

    dataset = tf.data.TFRecordDataset(args.scenario_tfrecord,
                                      compression_type='')
    with descr_out_path.open("wb") as descr_file, \
         idinfo_out_path.open("wb") as idinfo_file:
        for str_rec in dataset:
            scenario_msg = Scenario()
            scenario_msg.ParseFromString(str_rec.numpy())
            write_shape_descriptors_to(scenario_msg, descr_file, idinfo_file)
            write_track_trajectories_to_files(scenario_msg, trajectory_out_dir)


if __name__ == '__main__':
    main()
