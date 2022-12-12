all: descriptors.so

run-tests: descriptors_test.x descriptors.so
	./descriptors_test.x
	python3 -m nose2

format-sources:
	clang-format-12 -i *.cc *.h
	yapf3 -i *.py

descriptors.o: descriptors.cc descriptors.h
	g++ -c -O3 -Wall -Werror -std=c++17 \
    descriptors.cc -lboost_iostreams -lglog -o $@

descriptors.so: descriptors_pybind.cc descriptors.o
	g++ -O3 -Wall -Werror -shared -std=c++17 -fPIC \
        $$(python3 -m pybind11 --includes) \
    descriptors_pybind.cc descriptors.o -lboost_iostreams -lglog -o $@

descriptors_test.x: descriptors_test.cc descriptors.o
	g++ $^ -lglog -lboost_iostreams -lgtest -pthread -o $@

clean:
	rm -f *.so *.o *.x
