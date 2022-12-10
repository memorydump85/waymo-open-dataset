# TESTING:
# docker build -t dump_trajectories . && docker run -v /home/rpradeep/.config/gcloud:/cred -eCLOUD_RUN_TASK_INDEX=1 dump_trajectories:latest

import io
import os
import pathlib
import multiprocessing

import glog
from google.api_core.retry import Retry
import google.auth as gauth
from google.oauth2 import service_account
import google.protobuf as pb
from google.cloud import storage
import numpy as np
import tensorflow as tf

from waymo_open_dataset.protos.scenario_pb2 import Scenario, Track


CLOUD_RUN_TASK_INDEX = int(os.environ["CLOUD_RUN_TASK_INDEX"])
CLOUD_RUN_TASK_COUNT = int(os.environ.get("CLOUD_RUN_TASK_COUNT", 0))
glog.info("Executing task %s/%s.", CLOUD_RUN_TASK_INDEX, CLOUD_RUN_TASK_COUNT or "N")

credentials, project = gauth.load_credentials_from_file('/cred/application_default_credentials.json')
storage_client = storage.Client(credentials=credentials, project=project)
bucket = storage_client.get_bucket('waymo-od-track-trajectories')


def get_input_blob_as_file() -> pathlib.Path:
    source_manifest = io.BytesIO()
    storage_client.download_blob_to_file("gs://waymo-od-track-trajectories/_source_manifest",
                                        source_manifest)
    source_manifest = source_manifest.getvalue().decode().strip().split("\n")
    src_blob_uri = source_manifest[CLOUD_RUN_TASK_INDEX]

    blob_filepath = pathlib.Path("/tmp") / src_blob_uri.rsplit('/', 1)[-1]
    with blob_filepath.open('wb') as f:
        storage_client.download_blob_to_file(src_blob_uri, f)
    return blob_filepath


def upload_blob_from_file(blob_name, file):
    blob = bucket.blob(blob_name)
    blob.upload_from_file(file, rewind=True, retry=Retry())


def main():
    workdir = pathlib.Path("/tmp") / f"task-{CLOUD_RUN_TASK_INDEX:06d}"
    workdir.mkdir(exist_ok=False, parents=True)

    dataset = tf.data.TFRecordDataset(get_input_blob_as_file(), compression_type='')
    blobnames_and_contentfile = []
    for str_rec in dataset:
        scenario_msg = Scenario()
        scenario_msg.ParseFromString(str_rec.numpy())
        tstamps = scenario_msg.timestamps_seconds
        for track in scenario_msg.tracks:
            txy = np.array([(t, s.center_x, s.center_y)
                    for t, s in zip(tstamps, track.states) if s.valid],
                    dtype=np.float32)
            txy_file = io.BytesIO()
            np.save(txy_file, txy)
            blob_name = (f"{scenario_msg.scenario_id}/" +
                        f"{track.id:06d}--{track.object_type:1d}.npy")
            blobnames_and_contentfile.append((blob_name, txy_file))

    with multiprocessing.pool.Pool(128) as p:
        p.starmap(upload_blob_from_file, blobnames_and_contentfile)

if __name__ == '__main__':
    main()
