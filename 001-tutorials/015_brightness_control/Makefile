##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/makefile-common

VPATH := ${srcdir}/generic

device ?= $(if $(filter 1,$(NPU2)),npu2,npu)
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

BRIGHTNESS_VAL = 255

targetname = dataflow_graph

aie_py_src=${targetname}.py
use_placed?=0

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
endif

.PHONY: all template clean

all: build/final_${IMAGE_WIDTH}.xclbin

build/aie2_lineBased_8b_${IMAGE_WIDTH}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${device} ${IMAGE_WIDTH} ${IMAGE_HEIGHT} ${BRIGHTNESS_VAL} > $@

build/aie_kernel.cc.o: aie_kernel.cc
	mkdir -p ${@D}
ifeq ($(device),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}
else ifeq ($(device),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}
else
	echo "Device type not supported"
endif
	
build/final_${IMAGE_WIDTH}.xclbin: build/aie2_lineBased_8b_${IMAGE_WIDTH}.mlir build/aie_kernel.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname} -DIMAGE_WIDTH=${IMAGE_WIDTH} -DIMAGE_HEIGHT=${IMAGE_HEIGHT} -D CMAKE_C_COMPILER=gcc-13 -D CMAKE_CXX_COMPILER=g++-13 
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final_${IMAGE_WIDTH}.xclbin
	${powershell} ./$< -x build/final_${IMAGE_WIDTH}.xclbin -i build/insts.bin -k MLIR_AIE

clean:
	rm -rf build _build ${targetname}.exe Output_Image.jpg
