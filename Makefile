.PHONY : build install-python clean clean-log format test-all

TYPE ?= release
BUILD_SHARED ?= OFF
FORMAT_ORIGIN ?=

CMAKE_OPT = -DCMAKE_BUILD_TYPE=$(TYPE) -DBUILD_SHARED=$(BUILD_SHARED)

build:
	mkdir -p build/$(TYPE)
	cd build/$(TYPE) && cmake $(CMAKE_OPT) ../.. && make -j

install-python: build
	cp build/$(TYPE)/src/07python_ffi/python_ffi*.so src/07python_ffi/src/refactor_graph
	pip install -e src/07python_ffi/

clean:
	rm -rf build

clean-log:
	rm -rf log

format:
	@python3 scripts/format.py $(FORMAT_ORIGIN)

test-all:
	./build/$(TYPE)/src/00common/common_test
	./build/$(TYPE)/src/01graph_topo/graph_topo_test
	./build/$(TYPE)/src/04frontend/frontend_test
	./build/$(TYPE)/src/05onnx/onnx_test
