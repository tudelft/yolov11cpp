



all: onnxruntime-linux-x64-1.23.2/include/onnxruntime_cxx_api.h
	mkdir -p build
	cd build && cmake .. && make -j8

clean:
	rm -rf build
	rm -rf ./onnxruntime-linux-x64-1.23.2/


onnxruntime-linux-x64-1.23.2/include/onnxruntime_cxx_api.h:
	tar -xvf onnxruntime-linux-x64-1.23.2.tgz