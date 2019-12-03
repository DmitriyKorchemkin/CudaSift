#ifndef NVJPEG_UTILS
#define NVJPEG_UTILS
#include <nvjpeg.h>
namespace cudasift {
void printNVJpegFailure(const nvjpegStatus_t &status, const char *file,
                        int line);

#define safeCallNVJpeg(expr)                                                   \
  {                                                                            \
    nvjpegStatus_t status = expr;                                              \
    if (status != NVJPEG_STATUS_SUCCESS) {                                     \
      printNVJpegFailure(status, __FILE__, __LINE__);                          \
    }                                                                          \
  }

struct NVJPEG {
  static nvjpegHandle_t GetHandle();
  nvjpegHandle_t handle() const;

private:
  NVJPEG();
  ~NVJPEG();
  nvjpegHandle_t nvhandle;
};

} // namespace cudasift

#endif
