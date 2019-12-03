#include "cudasift/nvjpegUtils.h"
#include <iostream>

namespace cudasift {
void printNVJpegFailure(const nvjpegStatus_t &status, const char *file,
                        int line) {
  switch (status) {
  case NVJPEG_STATUS_SUCCESS:
    return;
  case NVJPEG_STATUS_NOT_INITIALIZED:
    std::cout << file << ':' << line << " nvJPEG: not initialized\n";
    break;
  case NVJPEG_STATUS_INVALID_PARAMETER:
    std::cout << file << ':' << line << " nvJPEG: invalid parameter\n";
    break;
  case NVJPEG_STATUS_BAD_JPEG:
    std::cout << file << ':' << line << " nvJPEG: bad jpeg\n";
    break;
  case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
    std::cout << file << ':' << line << " nvJPEG: not supported jpeg\n";
    break;
  case NVJPEG_STATUS_ALLOCATOR_FAILURE:
    std::cout << file << ':' << line << " nvJPEG: allocator failure\n";
    break;
  case NVJPEG_STATUS_EXECUTION_FAILED:
    std::cout << file << ':' << line << " nvJPEG: execution failed\n";
    break;
  case NVJPEG_STATUS_ARCH_MISMATCH:
    std::cout << file << ':' << line << " nvJPEG: arch mismatch\n";
    break;
  case NVJPEG_STATUS_INTERNAL_ERROR:
    std::cout << file << ':' << line << " nvJPEG: internal error\n";
    break;
  default:
    std::cout << file << ':' << line << " nvJPEG: unknow error: " << status
              << '\n';
    break;
  }
  exit(-100);
}
nvjpegHandle_t NVJPEG::GetHandle() {
  static NVJPEG nvjpeg;
  return nvjpeg.handle();
}
nvjpegHandle_t NVJPEG::handle() const { return nvhandle; }
NVJPEG::NVJPEG() {
  safeCallNVJpeg(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, nullptr, &nvhandle));
}
NVJPEG::~NVJPEG() { safeCallNVJpeg(nvjpegDestroy(nvhandle)); }

} // namespace cudasift
