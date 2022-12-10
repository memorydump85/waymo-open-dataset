FROM ubuntu:20.04

RUN apt-get update \
    && apt-get install --no-install-recommends -y python3-pip \
    && apt-get clean
RUN pip3 install --no-cache-dir \
        glog==0.3.1 numpy==1.19.5 \
        tensorflow==2.6.0 waymo-open-dataset-tf-2-6-0==1.4.9 protobuf==3.19.0
RUN pip3 install --no-cache-dir \
        google-cloud-storage==2.6.0 flask protobuf==3.19.0 scipy==1.9.3

COPY dump_trajectories_job.py /code/
# Ugly and unsafe. But it is not clear how to access external buckets
# using service account credentials.
COPY application_default_credentials.json /cred/application_default_credentials.json
CMD python3 /code/dump_trajectories_job.py
