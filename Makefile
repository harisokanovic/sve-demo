all: build/sve-demo build/sve-demo.asm

# -O0 = No optimizations.
# -O1 = Basic optimizations.
# -O2 = Optimize even more. GCC performs nearly all supported optimizations that do not involve a space-speed tradeoff. 
# -O3 = Unroll/peel loops. May bloat code.
# -Ofast = Disregard strict standards compliance. -Ofast enables all -O3 optimizations. It also enables optimizations that are not valid for all standard-compliant programs.

build/sve-demo: sve-demo.cpp Runtime.cpp
	g++ -march=armv8-a+sve -g -O2 -I. -o $@ $^ -lpthread

build/sve-demo.asm: build/sve-demo
	objdump -S $< | c++filt > $@

clean:
	rm -Rf build/*
