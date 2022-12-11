all: descriptors.so

format-sources:
	clang-format-12 -i *.cc
	yapf3 -i *.py

descriptors.so: descriptors.cc
	g++ -O3 -Wall -shared -std=c++17 -fPIC $$(python3 -m pybind11 --includes) $< -o $@
