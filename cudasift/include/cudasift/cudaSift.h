#ifndef CUDASIFT_H
#define CUDASIFT_H

#include "cudasift/SiftDetector.hpp"
#include "cudasift/cudaImage.h"
#include <cstdint>
#include <cuda_runtime_api.h>
#include <mutex>

namespace cudasift {

struct SiftData {
  int numPts; // Number of available Sift points
  int maxPts; // Number of allocated Sift points

  SiftData(const int maxPts = 0);
  SiftData(SiftData &&other);
  ~SiftData();
  void allocate(int num = 32768, bool host = false, bool dev = true,
                int device = 0);
  SiftPoint *hostPtr();
  void transferAsync(bool toHost, void *stream);
  void transfer(bool toHost, void *stream, bool sync);
  int deviceId = -1;
#ifdef MANAGEDMEM
  SiftPoint *m_data = nullptr; // Managed data
#else
  SiftPoint *h_data = nullptr; // Host (CPU) data
  SiftPoint *d_data = nullptr; // Device (GPU) data
#endif
private:
  SiftData(const SiftData &) = delete;
};

void InitCuda(int devNum = 0);
float *AllocSiftTempMemory(int width, int height, int numOctaves,
                           bool scaleUp = false);
void FreeSiftTempMemory(float *memoryTmp);
void PrintSiftData(SiftData &data);
double MatchSiftData(SiftData &data1, SiftData &data2);

struct DetectorConfigDevice {
  DetectorConfigDevice(int, void **, cudaStream_t);
  ~DetectorConfigDevice();

  mutable uint32_t pointCounter[8 * 2 + 1];
  uint32_t maxNumPoints;

  const float *scaleDownKernel;
  const float *lowPassKernel;
  const float *laplaceKernel;

private:
  DetectorConfigDevice(const DetectorConfigDevice &) = delete;
};

struct DetectorConfigHost {
  DetectorConfigHost(int, cudaStream_t);
  ~DetectorConfigHost();

  DetectorConfigDevice *dev;
  float oldScaleLowPass;
  float oldVarianceScaleDown;

  float *scaleDownKernel;
  float *lowPassKernel;
  float *laplaceKernel;
  uint32_t *pointCounter;

private:
  DetectorConfigHost(const DetectorConfigHost &) = delete;
};

struct SiftDetectorImpl {
  static const int MIN_ALIGNMENT = 128;
  SiftDetectorImpl(const SiftParams &params = SiftParams(), int device = 0,
                   cudaStream_t stream = nullptr);

  ~SiftDetectorImpl();
  void ExtractSift(SiftData &siftData, CudaImage &img, bool sync = true);

private:
  double ScaleDown(CudaImage &res, CudaImage &src, float variance);
  int ExtractSiftLoop(SiftData &siftData, CudaImage &img, int numOctaves,
                      float blur, float lowestScale, float subsampling,
                      float *memorySub);

  void ExtractSiftOctave(SiftData &siftData, CudaImage &img, int octave,
                         float lowestScale, float subsampling);
  double ComputeOrientations(cudaTextureObject_t texObj, CudaImage &src,
                             SiftData &siftData, int octave);
  double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData,
                                float subsampling, int octave);
  double LaplaceMulti(cudaTextureObject_t texObj, CudaImage &baseImage,
                      CudaImage *results, int octave);
  double FindPointsMulti(CudaImage *sources, SiftData &siftData,
                         float edgeLimit, float factor, float lowestScale,
                         float subsampling, int octave);
  double ScaleUp(CudaImage &res, CudaImage &src);
  double RescalePositions(SiftData &siftData, float scale);
  void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel);

  double LowPass(CudaImage &res, CudaImage &src, float scale);
  void realloc(int h, int w);
  SiftDetectorImpl(const SiftDetectorImpl &) = delete;
  DetectorConfigHost configHost;
  float *memoryTmp = nullptr;
  int memoryAlloc = 0;
  CudaImage scaledUp, lowImg;

  SiftParams params;
  cudaStream_t stream;
  int device;
  DescriptorNormalizerData *p_normalizer_d;
};

} // namespace cudasift

#endif
