all: process_scenario_tracks.docker-image.tgz

format-sources:
	# clang-format-12 -i *.cc *.h
	yapf3 -i *.py

process_scenario_tracks.docker-image.tgz:
	docker build -t process_scenario_tracks:latest .
	docker save process_scenario_tracks:latest | gzip > $@

dockerrun:
	docker run -v ${PWD}:/output process_scenario_tracks:latest \
        python3 /code/process_scenario_tracks.py -o /output/ /output/training_20s.tfrecord-00000-of-01000

clean:
	rm -f *.so *.o *.x
