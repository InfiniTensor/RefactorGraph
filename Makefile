.PHONY : build install-python clean clean-log format test-all

TYPE ?= debug
FORMAT_ORIGIN ?=
CUDA ?= OFF

CMAKE_OPT = -DCMAKE_BUILD_TYPE=$(TYPE) -DUSE_CUDA=$(CUDA)

build:
	mkdir -p build/$(TYPE)
	cd build/$(TYPE) && cmake $(CMAKE_OPT) ../.. && make -j

install-python: build
	cp build/$(TYPE)/src/09python_ffi/python_ffi*.so src/09python_ffi/src/refactor_graph
	pip install -e src/09python_ffi/

clean:
	rm -rf build

clean-log:
	rm -rf log

format:
	@python3 scripts/format.py $(FORMAT_ORIGIN)

test:
	cd build/$(TYPE) && make test -j
