.PHONY : build install-python reconfig clean clean-log format test

TYPE ?= debug
CUDA ?= OFF

FORMAT_ORIGIN ?=

build:
	mkdir -p build
	cmake -DCMAKE_BUILD_TYPE=$(TYPE) -DUSE_CUDA=$(CUDA) -Bbuild
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

format:
	@python3 scripts/format.py $(FORMAT_ORIGIN)
