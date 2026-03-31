
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "OpenCVUtils.h"

#define SRS_SHIFT 6
#define SCALE_MUL 64 

constexpr int HEIGHT = IMAGE_HEIGHT;
constexpr int WIDTH = IMAGE_WIDTH;
constexpr int IN_SIZE_A = WIDTH*HEIGHT;
constexpr int IN_SIZE_B = (WIDTH+64)*HEIGHT;
constexpr int OUT_SIZE_A = 64*HEIGHT;
constexpr int OUT_SIZE_B = WIDTH*HEIGHT;

constexpr uint64_t testImageWidth = WIDTH;
constexpr uint64_t testImageHeight = HEIGHT;
constexpr uint64_t testImageSize = testImageWidth * testImageHeight;

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  cxxopts::Options options("join_L2");
  test_utils::add_default_options(options);

  options.add_options()("image,p", "the input image",
                        cxxopts::value<std::string>())(
      "outfile,o", "the output image",
      cxxopts::value<std::string>()->default_value("Output_Normalized.jpg"));

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);
  cv::Mat inImageGray;
  cv::String fileIn;
  if (vm.count("image")) {
    fileIn =
        vm["image"]
            .as<std::
                    string>(); 
    initializeSingleGrayImageTest(fileIn, inImageGray);
  } else {
    fileIn = "RANDOM";
    inImageGray = cv::Mat(testImageHeight, testImageWidth, CV_8UC1);
    cv::randu(inImageGray, cv::Scalar(0), cv::Scalar(255));
  }

  cv::String fileOut =
      vm["outfile"].as<std::string>(); //"Output_Transpose.jpg";
  printf("Load input image %s and run Normalization\n", fileIn.c_str());

  cv::resize(inImageGray, inImageGray,
             cv::Size(testImageWidth, testImageHeight));

  cv::Mat outImageTest(testImageHeight, testImageWidth, CV_8UC1);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 std::cout << "Name: " << name << std::endl;
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";

  device.register_xclbin(xclbin);

  // get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_inA = xrt::bo(device, inImageGray.total() * inImageGray.elemSize(),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_inB = xrt::bo(device, IN_SIZE_B * sizeof(uint8_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out1 = xrt::bo(device, OUT_SIZE_A * sizeof(uint8_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out2 =
      xrt::bo(device, (outImageTest.total() * outImageTest.elemSize()),
              XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));


  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  uint8_t *bufInA = bo_inA.map<uint8_t *>();
  std::vector<uint8_t> srcVecA;
  for (int i = 0; i < IN_SIZE_A; i++)
    srcVecA.push_back(static_cast<uint8_t>(i % 256));
  memcpy(bufInA, inImageGray.data,
         (inImageGray.total() * inImageGray.elemSize()));

  uint8_t *bufInB = bo_inB.map<uint8_t *>();
  std::vector<uint8_t> srcVecB;
  for (int i = 0; i < IN_SIZE_B; i++)
    srcVecB.push_back(static_cast<uint8_t>(i % 256));
  memcpy(bufInB, srcVecB.data(), (srcVecB.size() * sizeof(uint8_t)));

  memcpy(outImageTest.data, bufInA,
         (outImageTest.total() * outImageTest.elemSize()));
		 
  cv::imwrite("image_in.bmp", outImageTest);

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  uint8_t *bufOut1 = bo_out1.map<uint8_t *>();
  uint8_t *bufOut2 = bo_out2.map<uint8_t *>();
  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_inA,bo_inB,bo_out1,bo_out2);
  run.wait();
  bo_out1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bo_out2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
#if 1

	uint8_t maxPix = 0;

    for (uint32_t i = 0; i < 64*HEIGHT; i++) {
		if(bufOut1[i] > maxPix)
			maxPix = bufOut1[i];
    }  
	std::cout << "Max val  output " << (int)maxPix << std::endl;
#endif
    std::cout <<""<<std::endl;
  /*Calculate scaling value*/
  uint16_t scaleVal = (uint16_t)(255*SCALE_MUL/((int32_t)maxPix));

 if(scaleVal > 255) {
scaleVal = 255;
}
  std::cout << "scaleVal  output " << (int)scaleVal << std::endl;
  memset(bufInB,0, (srcVecB.size() * sizeof(uint8_t)));
  for (uint32_t i = 0; i < HEIGHT; i++){
     bufInB[i*(WIDTH+64)] = (uint8_t)scaleVal;
	 memcpy(&bufInB[i*(WIDTH+64)+64],&bufInA[i*(WIDTH)], (WIDTH * sizeof(uint8_t)));
  }
 
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  run = kernel(opcode, bo_instr, instr_v.size(), bo_inA,bo_inB,bo_out1,bo_out2);
  run.wait();
  bo_out1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bo_out2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);


  uint8_t *bufOut = bo_out2.map<uint8_t *>();

  memcpy(outImageTest.data, bufOut,
         (outImageTest.total() * outImageTest.elemSize()));

  cv::imwrite(fileOut, outImageTest);

  printf("\nImage Normalization done!\n");
  return 0;
}

