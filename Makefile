.PHONY : build clean format

TYPE ?= release
BUILD_SHARED ?= OFF
FORMAT_ORIGIN ?=

CMAKE_OPT = -DCMAKE_BUILD_TYPE=$(TYPE) -DBUILD_SHARED=$(BUILD_SHARED)

build:
	mkdir -p build/$(TYPE)
	cd build/$(TYPE) && cmake $(CMAKE_OPT) ../.. && make -j

clean:
	rm -rf build

format:
	@python3 scripts/format.py $(FORMAT_ORIGIN)
