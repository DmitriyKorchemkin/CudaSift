#include "cudasift/SiftDetector.hpp"
#include "cudasift/cudaImage.h"
#include "cudasift/cudaSift.h"
#include "cudasift/cudautils.h"
#include "cudasift/nvjpegUtils.h"
#include <turbojpeg.h>

namespace cudasift {

#define BLOCK_CONVERT 8
#define BLOCK_CONVERT_W 32
#define BLOCK_CONVERT_H 4

void __global__ bgr_to_float(const uint8_t *bgr, float *fp32, int width,
                             int height, int stride_bytes, int stride_out) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int x = blockIdx.x * BLOCK_CONVERT + tx;
  const int y = blockIdx.y * BLOCK_CONVERT + ty;
  if (x >= width || y >= height)
    return;
  const int id_bgr = stride_bytes * y + x * 3;
  const int id_float = stride_out * y + x;

  float b = bgr[id_bgr];
  float g = bgr[id_bgr + 1];
  float r = bgr[id_bgr + 2];

  float gray = 0.299f * r + 0.587 * g + 0.114f * b;
  fp32[id_float] = gray;
}

void __global__ u8_to_float(const uint8_t *u8, float *fp32, int width,
                            int height, int stride_bytes, int stride_out) {
  const int ty = threadIdx.y;

  const int xs = blockIdx.x * BLOCK_CONVERT_W;
  const int y = blockIdx.y * BLOCK_CONVERT_H + ty;
  if (xs >= width || y >= height)
    return;
  int x = xs;
  int id_u8 = stride_bytes * y + xs;
  int id_float = stride_out * y + xs;
  for (; x < xs + BLOCK_CONVERT_W && x < width; ++x)
    fp32[id_float++] = u8[id_u8++];
}

void CudaSift::DetectorContext::processBgrU8(int w, int h, int stride_bytes) {
  dim3 blocks(iDivUp(w, BLOCK_CONVERT), iDivUp(h, BLOCK_CONVERT));
  dim3 threads(BLOCK_CONVERT, BLOCK_CONVERT);
  int stride_out = iAlignUp(w, SiftDetectorImpl::MIN_ALIGNMENT);
  cudasift::bgr_to_float<<<blocks, threads, 0, (cudaStream_t)stream>>>(
      u8_device, float_device, w, h, stride_bytes, stride_out);
  safeCall(cudaStreamSynchronize((cudaStream_t)stream));
  processFP32(w, h, stride_out * sizeof(float));
}

void CudaSift::DetectorContext::processU8(int w, int h, int stride_bytes) {
  dim3 blocks(iDivUp(w, BLOCK_CONVERT_W), iDivUp(h, BLOCK_CONVERT_H));
  dim3 threads(BLOCK_CONVERT_W, BLOCK_CONVERT_H);
  int stride_out = iAlignUp(w, SiftDetectorImpl::MIN_ALIGNMENT);
  cudasift::u8_to_float<<<blocks, threads, 0, (cudaStream_t)stream>>>(
      u8_device, float_device, w, h, stride_bytes, stride_out);
  processFP32(w, h, stride_out * sizeof(float));
}

void CudaSift::DetectorContext::processFP32(int w, int h, int stride) {
  safeCall(cudaStreamSynchronize((cudaStream_t)stream));
  checkMsg("Failed already");
  CudaImage image1, image2;
  image1.Allocate(w, h, stride / sizeof(float), false, float_device, nullptr);
  detector->ExtractSift(*siftData, image1, !collectCovariance);
  if (collectCovariance && siftData->numPts) {
    // need to aggregate covariance & sync after
    covariance->addDescriptorsDevice(siftData->d_data->data, siftData->numPts,
                                     sizeof(SiftPoint) / sizeof(float));
    safeCall(cudaStreamSynchronize((cudaStream_t)stream));
  }
}

void CudaSift::DetectorContext::processJPEG(const uint8_t *data, int w, int h,
                                            int size_bytes) {
  nvjpegOutputFormat_t fmt = NVJPEG_OUTPUT_Y;
  nvjpegImage_t dst;
  memset(&dst, 0, sizeof(nvjpegImage_t));
  dst.channel[0] = u8_device;
  dst.pitch[0] = w;
  auto status =
      nvjpegDecode(NVJPEG::GetHandle(), (nvjpegJpegState_t)nvjpegState, data,
                   size_bytes, fmt, &dst, (cudaStream_t)stream);
  // nvjpeg failed, retry with turbojpeg
  if (status == NVJPEG_STATUS_JPEG_NOT_SUPPORTED) {
    auto tjstatus = tjDecompress2(tjState, data, size_bytes, u8_pinned, w, 0, h,
                                  TJPF_GRAY, TJFLAG_FASTDCT);
    if (tjstatus && tjGetErrorCode(tjState) == TJERR_FATAL) {
      throw std::runtime_error(
          "Failed to decompress by both nvJPEG & turbo-jpeg");
    }
    safeCall(cudaMemcpyAsync(u8_device, u8_pinned, w * h,
                             cudaMemcpyHostToDevice, (cudaStream_t)stream));
  } else {
    printNVJpegFailure(status, __FILE__, __LINE__);
  }

  processU8(w, h, w);
}

} // namespace cudasift
