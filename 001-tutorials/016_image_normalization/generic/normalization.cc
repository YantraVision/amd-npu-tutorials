//===- normalization.cc -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>
#include <aie_kernels/aie_kernel_utils.h>

#define SRS_SHIFT 6

template <typename T, int N>
__attribute__((noinline)) void normKernel_func(T *restrict in, T *restrict out,
                                               const int32_t height,
                                               const int32_t width) {
  event0();

  uint8_t *outPtr = (uint8_t *)out;
  uint8_t *inPtr = (uint8_t *)in;
  
  uint8_t *header = (uint8_t * )in;
  inPtr+= 64;
  

  aie::vector<uint8, 64> temp;
  aie::vector<uint8, 64> scale = aie::broadcast<uint8,64>(header[0]);

  aie::accum<acc32, 64> accval;
  aie::vector<uint8, 64> tempS;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int j = 0; j < (height * width); j += N) // Nx samples per loop
  {
	temp = aie::load_v<64>(inPtr);
	accval = aie::mul(temp, scale);
	tempS =  accval.to_vector<uint8_t>(SRS_SHIFT);
	aie::store_v(outPtr, tempS );
	inPtr+= 64;
	outPtr+= 64;
  }

  event1();
}

template <typename T, int N>
__attribute__((noinline)) void maxKernel_func(T *restrict in, T *restrict out,
                                               const int32_t height,
                                               const int32_t width) {
  event0();

  v64uint8 *restrict outPtr = (v64uint8 *)out;
  v64uint8 *restrict inPtr = (v64uint8 *)in;
  
  aie::vector<uint8, 64> temp;
  uint8 *outSPtr = (uint8 *)out;
  uint8 *inSPtr = (uint8 *)in;
  aie::vector<uint8, 64> maxVal = aie::broadcast<uint8,64>(0);

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int j = 0; j < (height * width); j += 64) // Nx samples per loop
  {
	temp = aie::load_v<64>( inSPtr );
	maxVal = aie::max(temp, maxVal); 
	aie::store_v(outSPtr, maxVal );
    inSPtr += 64;
  }

  event1();
}

extern "C" {

void normKernel(int32_t *in, int32_t *out, int32_t lineWidth) {
  normKernel_func<int32_t, 16>(in, out, 1, lineWidth);
}
void maxKernel(int32_t *in, int32_t *out, int32_t lineWidth) {
  maxKernel_func<int32_t, 16>(in, out, 1, lineWidth);
}

} // extern "C"
