import argparse
import pathlib
import shutil
import subprocess

import glog
import google.protobuf as pb
import numpy as np
import tensorflow as tf

from waymo_open_dataset.protos.scenario_pb2 import Scenario, Track


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tfrecord_gs_uri", type=str)
    args = parser.parse_args()

    workdir = pathlib.Path("/tmp/dump_trajectories.work")
    if workdir.exists(): shutil.rmtree(workdir)
    workdir.mkdir(exist_ok=False, parents=True)

    downloaded_filepath = workdir / "input.tfrecord"
    subprocess.check_call(["gsutil", "cp", args.tfrecord_gs_uri,
                           str(downloaded_filepath)])

    dataset = tf.data.TFRecordDataset(downloaded_filepath, compression_type='')
    for str_rec in dataset:
        scenario_msg = Scenario()
        scenario_msg.ParseFromString(str_rec.numpy())
        tstamps = scenario_msg.timestamps_seconds
        outdir = workdir / scenario_msg.scenario_id
        outdir.mkdir(exist_ok=True, parents=True)
        glog.info("Writing %s", outdir)
        for track in scenario_msg.tracks:
            txy = np.array([(t, s.center_x, s.center_y)
                    for t, s in zip(tstamps, track.states) if s.valid],
                    dtype=np.float32)
            outfile_name = (outdir
                            / f"{track.id:06d}--{track.object_type:1d}.npy")
            np.save(outfile_name, txy, allow_pickle=False)

    subprocess.check_call(["gsutil", "-m", "rsync", "-r", "-x", "input.tfrecord",
                           workdir, "gs://waymo-od-track-trajectories/"])

if __name__ == '__main__':
    main()
