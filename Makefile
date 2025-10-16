.PHONY: configure build run

configure:
	cmake --preset debug

build:
	cmake --build build-debug --parallel

run:
	./build-debug/testbed/testbed