#ifndef CUDASIFT_H
#define CUDASIFT_H

#include "cudasift/SiftDetector.hpp"
#include "cudasift/cudaImage.h"
#include <cstdint>
#include <cuda_runtime_api.h>

namespace cudasift {

typedef struct {
  float xpos;
  float ypos;
  float scale;
  float sharpness;
  float edgeness;
  float orientation;
  float score;
  float ambiguity;
  int match;
  float match_xpos;
  float match_ypos;
  float match_error;
  float subsampling;
  float empty[3];
  float data[128];
} SiftPoint;

typedef struct {
  int numPts; // Number of available Sift points
  int maxPts; // Number of allocated Sift points
#ifdef MANAGEDMEM
  SiftPoint *m_data; // Managed data
#else
  SiftPoint *h_data; // Host (CPU) data
  SiftPoint *d_data; // Device (GPU) data
#endif

} SiftData;

void InitCuda(int devNum = 0);
float *AllocSiftTempMemory(int width, int height, int numOctaves,
                           bool scaleUp = false);
void FreeSiftTempMemory(float *memoryTmp);
void InitSiftData(SiftData &data, int num = 1024, bool host = false,
                  bool dev = true);
void FreeSiftData(SiftData &data);
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
  SiftDetectorImpl(const SiftParams &params = SiftParams(), int device = 0,
                   cudaStream_t stream = nullptr);

  ~SiftDetectorImpl();
  void ExtractSift(SiftData &siftData, CudaImage &img);

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
