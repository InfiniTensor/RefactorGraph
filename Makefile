.PHONY : build clean format

TYPE ?= release
FORMAT_ORIGIN ?=

CMAKE_OPT = -DCMAKE_BUILD_TYPE=$(TYPE)

build:
	mkdir -p build/$(TYPE)
	cd build/$(TYPE) && cmake $(CMAKE_OPT) ../.. && make -j

clean:
	rm -rf build

format:
	@python3 scripts/format.py $(FORMAT_ORIGIN)
