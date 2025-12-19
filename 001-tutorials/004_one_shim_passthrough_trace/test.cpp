

#include "xrt_test_wrapper.h"
#include <cstdint>

//*****************************************************************************
// Modify this section to customize buffer datatypes, initialization functions,
// and verify function. The other place to reconfigure your design is the
// Makefile.
//*****************************************************************************

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
// ------------------------------------------------------
// Configure this to match your buffer data type
// ------------------------------------------------------
using DATATYPE_IN1 = std::int32_t;
using DATATYPE_OUT = std::int32_t;
#endif

// Initialize Input buffer 1
void initialize_bufIn1(DATATYPE_IN1 *bufIn1, int SIZE) {
  for (int i = 0; i < SIZE; i++)
    bufIn1[i] = i;
}

// Initialize Output buffer
void initialize_bufOut(DATATYPE_OUT *bufOut, int SIZE) {
  memset(bufOut, 0, SIZE);
}

// Functional correctness verifyer
int verify_passthrough_kernel(DATATYPE_IN1 *bufIn1, DATATYPE_OUT *bufOut,
                              int SIZE, int verbosity) {
  int errors = 0;

  for (int i = 0; i < 1024; i++) {
    int32_t ref = bufIn1[i];
    int32_t test = bufOut[i];
    std::cout << "Data miss match " << test << " " << ref << std::endl;
    if (test != ref) {
      if (verbosity >= 1)
        std::cout << "Error in output " << test << " != " << ref << std::endl;
      errors++;
    } else {
      if (verbosity >= 1)
        std::cout << "Correct output " << test << " == " << ref << std::endl;
    }
  }
  return errors;
}

//*****************************************************************************
// Should not need to modify below section
//*****************************************************************************

int main(int argc, const char *argv[]) {

  constexpr int IN1_VOLUME = IN1_SIZE / sizeof(DATATYPE_IN1);
  constexpr int OUT_VOLUME = OUT_SIZE / sizeof(DATATYPE_OUT);

  args myargs = parse_args(argc, argv);

  int res = setup_and_run_aie<DATATYPE_IN1, DATATYPE_OUT, initialize_bufIn1,
                              initialize_bufOut, verify_passthrough_kernel>(
      IN1_VOLUME, OUT_VOLUME, myargs);
  return res;
}
