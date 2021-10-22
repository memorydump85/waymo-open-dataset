FROM tensorflow/tensorflow:latest-py3

RUN apt-get update && \
    apt-get install -y git build-essential wget vim findutils curl wget \
                       pkg-config zip g++ zlib1g-dev unzip python3 python3-pip

WORKDIR /waymo-od-install

RUN wget https://github.com/bazelbuild/bazel/releases/download/0.28.0/bazel-0.28.0-installer-linux-x86_64.sh && \
    bash ./bazel-0.28.0-installer-linux-x86_64.sh

ADD ./ /waymo-od-install/waymo-od

RUN cd /waymo-od-install/waymo-od && \
    bash ./configure.sh && \
    bash bazel query ... | xargs bazel build -c opt && \
    bash bazel query 'kind(".*_test rule", ...)' | xargs bazel test -c opt ... && \
    cp bazel-genfiles/waymo_open_dataset/*_pb2.py waymo_open_dataset/

RUN pip install --upgrade pip matplotlib scikit-learn

ENV PYTHONPATH=/waymo-od-install/waymo-od
ENV MPLCONFIGDIR=/tmp/matplotlibconfig
WORKDIR /code/waymo-od