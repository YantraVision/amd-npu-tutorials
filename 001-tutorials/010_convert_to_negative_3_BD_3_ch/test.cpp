#include "cxxopts.hpp"
#include "test_utils.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "OpenCVUtils.h"
#include "cxxopts.hpp"
#include "test_utils.h"

constexpr int channels = 4;
constexpr uint64_t testImageWidth = IMAGE_WIDTH;
constexpr uint64_t testImageHeight = IMAGE_HEIGHT;
constexpr uint64_t testImageSize = testImageWidth * testImageHeight;
constexpr int trace_size = 131072;
bool enable_ctrl_pkts = true;
constexpr int WIDTH  = 1920;
constexpr int HEIGHT = 1080;
constexpr size_t STEP = WIDTH;          // bytes per channel per row
constexpr size_t ROW_BYTES = 3 * STEP;  // R + G + B

struct args {
  int verbosity;
  int do_verify;
  int n_iterations;
  int n_warmup_iterations;
  int trace_size;
  std::string instr;
  std::string xclbin;
  std::string kernel;
  std::string trace_file;
};

struct args parse_args(int argc, const char *argv[]) {
  // ------------------------------------------------------
  // Parse program arguments
  // ------------------------------------------------------
  cxxopts::Options options("XRT Test Wrapper");
  cxxopts::ParseResult vm;
  test_utils::add_default_options(options);

  struct args myargs;

  test_utils::parse_options(argc, argv, options, vm);
  myargs.verbosity = vm["verbosity"].as<int>();
  myargs.do_verify = vm["verify"].as<bool>();
  myargs.n_iterations = vm["iters"].as<int>();
  myargs.n_warmup_iterations = vm["warmup"].as<int>();
  myargs.trace_size = vm["trace_sz"].as<int>();
  myargs.instr = vm["instr"].as<std::string>();
  myargs.xclbin = vm["xclbin"].as<std::string>();
  myargs.kernel = vm["kernel"].as<std::string>();
  myargs.trace_file = vm["trace_file"].as<std::string>();

  return myargs;
}

uint32_t getParity(uint32_t n) {
  int count = 0;
  while (n > 0) {
    if (n & 1) { // Check if the least significant bit is 1
      count++;
    }
    n >>= 1; // Right shift to check the next bit
  }
  return (count % 2 == 0) ? 0 : 1; // 0 for even parity, 1 for odd parity
}

uint32_t create_ctrl_pkt(int operation, int beats, int addr,
                         int ctrl_pkt_read_id = 28) {
  uint32_t ctrl_pkt = ((ctrl_pkt_read_id & 0xFF) << 24) |
                      ((operation & 0x3) << 22) | ((beats & 0x3) << 20) |
                      (addr & 0x7FFFF);
  ctrl_pkt |= (0x1 ^ getParity(ctrl_pkt)) << 31;
  return ctrl_pkt;
}

int main(int argc, const char *argv[]) {
  struct args myargs;
  // Program arguments parsing
  cxxopts::Options options("conv_to_negative");
  test_utils::add_default_options(options);
  options.add_options()("image,p", "the input image",
                        cxxopts::value<std::string>())(
      "outfile,o", "the output image",
      cxxopts::value<std::string>()->default_value("Output_Negative.jpg"));
  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  // Read the input image or generate random one if no input file argument
  // provided
  cv::Mat inImageRGB;
  cv::Mat inImageGray;
  cv::String fileIn;
  if (vm.count("image")) {
      fileIn = vm["image"].as<std::string>();

      // Load image as RGB (actually BGR in OpenCV)
      inImageRGB = cv::imread(fileIn, cv::IMREAD_COLOR);

      if (inImageRGB.empty()) {
          std::cerr << "ERROR: Could not load image: " << fileIn << std::endl;
          return -1;
      }

    // Resize to required dimensions
    cv::resize(inImageRGB, inImageRGB,
               cv::Size(testImageWidth, testImageHeight));  
  } else {
    fileIn = "RANDOM";
    inImageGray = cv::Mat(testImageHeight, testImageWidth, CV_8UC1);
    cv::randu(inImageGray, cv::Scalar(0), cv::Scalar(255));
  }
  
  cv::String fileOut =
      vm["outfile"].as<std::string>(); //"Output_Negative.jpg";
  printf("Load input image %s and run Negative Conversion\n", fileIn.c_str());

  cv::Mat outImageRGB(HEIGHT, WIDTH, CV_8UC3);
  cv::Mat outImageReference = inImageRGB.clone();
  cv::Mat outImageTest(testImageHeight, testImageWidth, CV_8UC1);

  // Load instruction sequence
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT context and load the kernel
  xrt::device device;
  xrt::kernel kernel;

  test_utils::init_xrt_load_kernel(device, kernel, verbosity,
                                   vm["xclbin"].as<std::string>(),
                                   vm["kernel"].as<std::string>());

					  //

  // set up the buffer objects
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_inA = xrt::bo(device,inImageRGB.total() * inImageRGB.elemSize(),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_inB = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out =
      xrt::bo(device, (outImageTest.total() * outImageTest.elemSize()),
              XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  // If we enable control packets, then this is the input xrt buffer for that.
  // Otherwise, this is a dummy placedholder buffer.
  auto bo_ctrlpkts =
      xrt::bo(device, 8, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
  int tmp_trace_size = (trace_size > 0) ? trace_size * 4 : 1;
  auto bo_trace = xrt::bo(device, tmp_trace_size, XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(7));


  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  uint8_t* bufInA = bo_inA.map<uint8_t*>();
  uint8_t* bufOut  = bo_out.map<uint8_t*>();

  for (int y = 0; y < HEIGHT; y++) {

    const cv::Vec3b* rowPtr = inImageRGB.ptr<cv::Vec3b>(y);

    // Pack one row as [R | G | B]
    for (int x = 0; x < WIDTH; x++) {
        bufInA[x]             = rowPtr[x][2]; // R (BGR order)
        bufInA[x + STEP]      = rowPtr[x][1]; // G
        bufInA[x + 2 * STEP]  = rowPtr[x][0]; // B
    }
  //}
    void *bufInstr = bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

    char *bufTrace = bo_trace.map<char *>();
    uint32_t *bufCtrlPkts = bo_ctrlpkts.map<uint32_t *>();

    if (trace_size > 0)
      memset(bufTrace, 0, trace_size);

    // Set control packet values
    if (trace_size > 0 && enable_ctrl_pkts) {
      bufCtrlPkts[0] = create_ctrl_pkt(1, 0, 0x32004); // core status
      bufCtrlPkts[1] = create_ctrl_pkt(1, 0, 0x320D8); // trace status
      if (verbosity >= 1) {
        std::cout << "bufCtrlPkts[0]:" << std::hex << bufCtrlPkts[0] << std::endl;
        std::cout << "bufCtrlPkts[1]:" << std::hex << bufCtrlPkts[1] << std::endl;
      }
    }

    // sync host to device memories
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    if (trace_size > 0) {
      bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      if (enable_ctrl_pkts)
        bo_ctrlpkts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    

    // Execute the kernel and wait to finish
    if (verbosity >= 1)
      std::cout << "Running Kernel.\n";
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_inB, bo_out, bo_ctrlpkts, bo_trace);
    run.wait();

    // Sync device to host memories
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    uint8_t* rPtr = bufOut;
    uint8_t* gPtr = bufOut + STEP;
    uint8_t* bPtr = bufOut + 2 * STEP;

    cv::Vec3b* outRow =
        outImageRGB.ptr<cv::Vec3b>(y);

    for (int x = 0; x < WIDTH; x++) {
        outRow[x][0] = bPtr[x]; // B
        outRow[x][1] = gPtr[x]; // G
        outRow[x][2] = rPtr[x]; // R
    }

    if (trace_size > 0)
        bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        test_utils::write_out_trace((char *)bufTrace, trace_size,
                                    "trace.txt");


    // Write out control packet outputs
    if (enable_ctrl_pkts) {
      uint32_t *ctrl_pkt_out =
          (uint32_t *)(((char *)bufTrace) + trace_size);
      if (verbosity >= 1) {
        std::cout << "ctrl_pkt_out[0]:" << std::hex << ctrl_pkt_out[0]
                  << std::endl;
        std::cout << "ctrl_pkt_out[1]:" << std::hex << ctrl_pkt_out[1]
                  << std::endl;
      }
      int col = (ctrl_pkt_out[0] >> 21) & 0x7F;
      int row = (ctrl_pkt_out[0] >> 16) & 0x1F;
      if ((ctrl_pkt_out[1] >> 8) == 3)
        std::cout << "WARNING: Trace overflow detected in tile(" << row << ","
                  << col << ". Trace results may be invalid." << std::endl;
    }

    // Store result in cv::Mat
    uint8_t *bufOut = bo_out.map<uint8_t *>();
    memcpy(outImageTest.data, bufOut,
           (outImageTest.total() * outImageTest.elemSize()));

  }
  cv::imwrite(fileOut, outImageRGB);

  printf("Image Conversion to Negative  done!\n");
  return 0;
}
