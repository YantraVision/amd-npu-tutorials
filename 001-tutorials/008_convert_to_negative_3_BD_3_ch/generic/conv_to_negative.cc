//===- passThrough.cc -------------------------------------------*- C++ -*-===//

//

// This file is licensed under the Apache License v2.0 with LLVM Exceptions.

// See https://llvm.org/LICENSE.txt for license information.

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//

// Copyright (C) 2022, Advanced Micro Devices, Inc.

//

//===----------------------------------------------------------------------===//



// #define __AIENGINE__ 1

#define NOCPP



#include <stdint.h>

#include <stdlib.h>



#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>



template <typename T, int N>

__attribute__((noinline)) void conv_to_negative(T *restrict in, T *restrict out,

                                               const int32_t height,

                                               const int32_t width) {

  event0();



  v64uint8 *restrict outPtr = (v64uint8 *)out;
  v64uint8 *restrict inPtr = (v64uint8 *)in;

  uint8 *outSPtr = (uint8 *)out;
  uint8 *inSPtr = (uint8 *)in;


  aie::vector<uint8, 64> tempR;
  aie::vector<uint8, 64> tempL = aie::broadcast<uint8,64>(255);;
  aie::vector<uint8, 64> tempS;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)

  for (int j = 0; j < (height * width); j += N) // Nx samples per loop
  {
#if 1
     tempR = aie::load_v<64>( inSPtr );
     tempS= aie::sub( tempL ,tempR);
     aie::store_v( outSPtr, tempS );

     outSPtr += 64;
     inSPtr += 64;
#else
    *outPtr++ = *inPtr++;
#endif
  }



  event1();

}



extern "C" {



#if BIT_WIDTH == 8



void conv_to_negativeLine(uint8_t *in, uint8_t *out, int32_t lineWidth) {

  conv_to_negative<uint8_t, 64>(in, out, 1, lineWidth);

}



void conv_to_negativeTile(uint8_t *in, uint8_t *out, int32_t tileHeight,

                     int32_t tileWidth) {

  conv_to_negative<uint8_t, 64>(in, out, tileHeight, tileWidth);

}





#endif



} // extern "C"
