all: _source_manifest dockerim

_source_manifest:
	gsutil ls -R gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario | grep tfrecord > $@
	gsutil cp _source_manifest gs://waymo-od-track-trajectories/_source_manifest

dockerim:
	docker build -t trajectory_dump_etl_job .
	docker tag trajectory_dump_etl_job:latest us-docker.pkg.dev/sensor-experiments/rep0/trajectory_dump_etl_job

dockerpush: dockerim
	docker push us-docker.pkg.dev/sensor-experiments/rep0/trajectory_dump_etl_job

dockerrun: _source_manifest dockerim
	docker run -it \
      -eCLOUD_RUN_TASK_INDEX=1  -eGOOGLE_CLOUD_PROJECT='sensor-experiments' \
      trajectory_dump_etl_job:latest
