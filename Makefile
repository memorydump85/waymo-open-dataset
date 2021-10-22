docker-run: docker-image
	# docker run -u `id -u`:`id -g` -dit --name waymo-od-devenv-container \
	docker run -dit --name waymo-od-devenv-container \
		-v /home/rpradeep/studio/waymo-open-dataset:/code/waymo-od \
		-v /tmp:/tmp \
		waymo-od-devenv bash

docker-image:
	docker build -f devenv/Dockerfile -t waymo-od-devenv .
