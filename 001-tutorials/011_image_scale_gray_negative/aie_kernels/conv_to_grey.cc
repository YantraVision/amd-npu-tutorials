
// #define __AIENGINE__ 1

#define NOCPP



#include <stdint.h>

#include <stdlib.h>



#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>

#define ROW_OFFSET 1920 
#define SRS_SHIFT 8 


template <typename T, int N>

__attribute__((noinline)) void conv_to_grey(T *restrict in, T *restrict out,

                                               const int32_t height,

                                               const int32_t width) {

  event0();



  v64uint8 *restrict outPtr = (v64uint8 *)out;
  v64uint8 *restrict inPtrR = (v64uint8 *)in;
  v64uint8 *restrict inPtrG = (v64uint8 *)(in + ROW_OFFSET);
  v64uint8 *restrict inPtrB = (v64uint8 *)(in + 2*ROW_OFFSET);

  uint8 *outSPtr = (uint8 *)out;
  uint8 *inSPtrR = (uint8 *)in;
  uint8 *inSPtrG = (uint8 *)(in + ROW_OFFSET);
  uint8 *inSPtrB = (uint8 *)(in + 2*ROW_OFFSET);


  aie::vector<uint8, 64> tempR;
  aie::vector<uint8, 64> tempG;
  aie::vector<uint8, 64> tempB;
  
  aie::vector<uint8, 64> scaleR = aie::broadcast<uint8,64>(77);
  aie::vector<uint8, 64> scaleG = aie::broadcast<uint8,64>(150);
  aie::vector<uint8, 64> scaleB = aie::broadcast<uint8,64>(29);
  
  aie::accum<acc32, 64> accGRY;
  aie::vector<uint8, 64> tempS;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)

  for (int j = 0; j < (height * width); j += N) // Nx samples per loop
  {
#if 1
     tempR = aie::load_v<64>( inSPtrR );
	 tempG = aie::load_v<64>( inSPtrG );
	 tempB = aie::load_v<64>( inSPtrB );
	 
	 accGRY = aie::mul(tempR, scaleR);
	 accGRY = aie::mac(accGRY, tempG, scaleG);
	 accGRY = aie::mac(accGRY, tempB, scaleB);
	 
	 tempS =  accGRY.to_vector<uint8_t>(SRS_SHIFT);
	 
     aie::store_v( outSPtr, tempS );

     outSPtr += 64;
     inSPtrR += 64;
	 inSPtrG += 64;
	 inSPtrB += 64;
#else
    *outPtr++ = *inPtr++;
#endif
  }



  event1();

}



extern "C" {



#if BIT_WIDTH == 8



void conv_to_greyLine(uint8_t *in, uint8_t *out, int32_t lineWidth) {

  conv_to_grey<uint8_t, 64>(in, out, 1, lineWidth);

}



void conv_to_greyTile(uint8_t *in, uint8_t *out, int32_t tileHeight,

                     int32_t tileWidth) {

  conv_to_grey<uint8_t, 64>(in, out, tileHeight, tileWidth);

}





#endif



} // extern "C"
