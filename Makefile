.PHONY : build install-python reconfig clean clean-log format test

TYPE ?= Debug
CUDA ?= OFF
KUNLUN ?= OFF

CMAKE_EXTRA =
# CMAKE_EXTRA += -DCMAKE_CXX_COMPILER=

build:
	mkdir -p build
	cmake -Bbuild -DCMAKE_BUILD_TYPE=$(TYPE) -DUSE_CUDA=$(CUDA) -DUSE_KUNLUN=$(KUNLUN) $(CMAKE_EXTRA)
	make -j -C build

install-python: build
	cp build/src/09python_ffi/python_ffi*.so src/09python_ffi/src/refactor_graph
	pip install -e src/09python_ffi/

reconfig:
	@rm -f build/CMakeCache.txt
	@rm -rf build/CMakeFiles
	@echo "configuration cache removed."

clean:
	rm -rf build

clean-log:
	rm -rf log

test:
	make test -j -Cbuild
