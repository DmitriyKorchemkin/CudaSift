//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

#include <cstdio>

#include "cudasift/cudaImage.h"
#include "cudasift/cudautils.h"

namespace cudasift {

int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }
int iDivDown(int a, int b) { return a / b; }
int iAlignUp(int a, int b) { return (a % b != 0) ? (a - a % b + b) : a; }
int iAlignDown(int a, int b) { return a - a % b; }

void CudaImage::Allocate(int w, int h, int p, bool host, float *devmem,
                         float *hostmem) {
  width = w;
  height = h;
  pitch = p;
  d_data = devmem;
  h_data = hostmem;
  t_data = NULL;
  if (devmem == NULL) {
    safeCall(cudaMallocPitch((void **)&d_data, (size_t *)&pitch,
                             (size_t)(sizeof(float) * width), (size_t)height));
    pitch /= sizeof(float);
    if (d_data == NULL)
      printf("Failed to allocate device data\n");
    d_internalAlloc = true;
  }
  if (host && hostmem == NULL) {
    safeCall(cudaHostAlloc((void **)&h_data, sizeof(float) * pitch * height,
                           cudaHostAllocPortable));
    h_internalAlloc = true;
  }
}

CudaImage::CudaImage()
    : width(0), height(0), d_data(NULL), h_data(NULL), t_data(NULL),
      d_internalAlloc(false), h_internalAlloc(false) {}

CudaImage::~CudaImage() {
  if (d_internalAlloc && d_data != NULL)
    safeCall(cudaFree(d_data));
  d_data = NULL;
  if (h_internalAlloc && h_data != NULL)
    safeCall(cudaFreeHost(h_data));
  h_data = NULL;
  if (t_data != NULL)
    safeCall(cudaFreeArray((cudaArray *)t_data));
  t_data = NULL;
}

double CudaImage::Download() {
  TimerGPU timer(0);
  int p = sizeof(float) * pitch;
  if (d_data != NULL && h_data != NULL)
    safeCall(cudaMemcpy2D(d_data, p, h_data, sizeof(float) * width,
                          sizeof(float) * width, height,
                          cudaMemcpyHostToDevice));
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("Download time =               %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

double CudaImage::Readback() {
  TimerGPU timer(0);
  int p = sizeof(float) * pitch;
  safeCall(cudaMemcpy2D(h_data, sizeof(float) * width, d_data, p,
                        sizeof(float) * width, height, cudaMemcpyDeviceToHost));
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("Readback time =               %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

double CudaImage::InitTexture() {
  TimerGPU timer(0);
  cudaChannelFormatDesc t_desc = cudaCreateChannelDesc<float>();
  safeCall(cudaMallocArray((cudaArray **)&t_data, &t_desc, pitch, height));
  if (t_data == NULL)
    printf("Failed to allocated texture data\n");
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("InitTexture time =            %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}


} // namespace cudasift
