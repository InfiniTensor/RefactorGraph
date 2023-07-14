.PHONY : build clean

TYPE ?= release

CMAKE_OPT = -DCMAKE_BUILD_TYPE=$(TYPE)

build:
	mkdir -p build/$(TYPE)
	cd build/$(TYPE) && cmake $(CMAKE_OPT) ../.. && make -j

clean:
	rm -rf build
