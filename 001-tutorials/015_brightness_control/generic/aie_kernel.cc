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

#define SRS_SHIFT 6

template <typename T, int N>

__attribute__((noinline)) void aie_kernel(T *restrict in, T *restrict out,

                                               const int32_t height,

                                               const int32_t width, int32_t BRIGHTNESS_VAL) {

   event0();

  v64uint8 *restrict outPtr = (v64uint8 *)out;
  v64uint8 *restrict inPtrR = (v64uint8 *)in;

  uint8 *outSPtr = (uint8 *)out;
  uint8 *inSPtr = (uint8 *)in;

  aie::vector<uint8, 64> temp;
  aie::vector<uint8, 64> scale = aie::broadcast<uint8,64>(BRIGHTNESS_VAL);

  aie::accum<acc32, 64> accval;
  aie::vector<uint8, 64> tempS;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)

  for (int j = 0; j < (height * width); j += N) // N x samples per loop
  {
#if 1
     temp = aie::load_v<64>( inSPtr );
	 
	 accval = aie::mul(temp, scale);

	 tempS =  accval.to_vector<uint8_t>(SRS_SHIFT);
	 
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


void aie_kernelLine(uint8_t *in, uint8_t *out, int32_t lineWidth, int32_t ctrl) {

  aie_kernel<uint8_t, 64>(in, out, 1, lineWidth, ctrl);

}


void aie_kernelTile(uint8_t *in, uint8_t *out, int32_t tileHeight,

                     int32_t tileWidth, int32_t ctrl) {

  aie_kernel<uint8_t, 64>(in, out, tileHeight, tileWidth, ctrl);

}

#endif


} // extern "C"
