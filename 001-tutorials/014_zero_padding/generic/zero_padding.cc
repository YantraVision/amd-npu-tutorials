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

#define NDIM (8)

void transpose8x8(uint8_t *src, uint8_t *dst)
{	
    int i, j;

    for (i = 0; i < NDIM; i++)
    {
        for (j = 0; j < NDIM; j++)
        {
            *(dst + j*NDIM + i) = *(src + i*NDIM + j);
        }
    }
}



template <typename T, int N>

__attribute__((noinline)) void zero_padding(T *restrict in, T *restrict out,
                                               const int32_t height,
                                               const int32_t width) {

  event0();
#if 0  //Transpose or passthrough
  uint8 *outSPtr = (uint8 *)out;
  uint8 *inSPtr = (uint8 *)in;

  for (int j = 0; j < (height * width); j += N) // Nx samples per loop
  {
#if 1 //Function or intrinsic
	transpose8x8(inSPtr, outSPtr);
#else
	v64uint8 *restrict outPtr = (v64uint8 *)outSPtr;
    v64uint8 *restrict inPtr = (v64uint8 *)inSPtr;
	*outPtr = aie::transpose(inPtr,NDIM,NDIM);
#endif	
    outSPtr += 64;
    inSPtr += 64;
  }
#else
  v64uint8 *restrict outPtr = (v64uint8 *)out;
  v64uint8 *restrict inPtr = (v64uint8 *)in;

  for (int j = 0; j < (height * width); j += N) // Nx samples per loop
  {
    *outPtr = *inPtr;
	outPtr++;
	inPtr++;
  }
#endif
  event1();

}



extern "C" {



#if BIT_WIDTH == 8



void zero_paddingLine(uint8_t *in, uint8_t *out, int32_t lineWidth) {

  zero_padding<uint8_t, 64>(in, out, 1, lineWidth);

}



void zero_paddingTile(uint8_t *in, uint8_t *out, int32_t tileHeight,

                     int32_t tileWidth) {

  zero_padding<uint8_t, 64>(in, out, tileHeight, tileWidth);

}





#endif



} // extern "C"
