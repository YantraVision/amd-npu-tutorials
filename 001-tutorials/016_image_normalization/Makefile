srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/makefile-common

VPATH := ${srcdir}/generic
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

targetname ?= img_normalization
aie_py_src=${targetname}.py
use_placed?=0

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
endif

all: build/final.xclbin

build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${devicename} ${IMAGE_WIDTH} ${IMAGE_HEIGHT} > $@


build/normalization.cc.o: generic/normalization.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -c ../generic/normalization.cc -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c ../generic/normalization.cc -o ${@F}
else
	echo "Device type not supported"
endif
.PHONY: inst/insts.bin
inst/insts.bin: ${srcdir}/${aie_py_src}
	rm -rf inst
	mkdir -p inst 
	python3 $< ${LENGTH} > inst/aie.mlir
	pushd inst && aiecc.py --aie-only-generate-npu --npu-insts-name=insts.bin aie.mlir && popd
	${powershell} ./build/${targetname}.exe -x build/final.xclbin -i inst/insts.bin -k MLIR_AIE -l ${LENGTH}

build/final.xclbin: build/aie.mlir build/normalization.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				--no-xchesscc --no-xbridge \
				--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -D CMAKE_C_COMPILER=gcc-13 -D CMAKE_CXX_COMPILER=g++-13 -DTARGET_NAME=${targetname} 
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final.xclbin
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE

clean:
	rm -rf build _build ${targetname}.exe
