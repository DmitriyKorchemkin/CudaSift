#ifndef CUDASIFTH_H
#define CUDASIFTH_H

#include "cudasift/cudaImage.h"
#include "cudasift/cudautils.h"

//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

namespace cudasift {

struct DetectorConfigDevice {
  DetectorConfigDevice(int, void **);
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
  DetectorConfigHost(int);
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

int ExtractSiftLoop(DetectorConfigHost *cfg, SiftData &siftData, CudaImage &img,
                    int numOctaves, double initBlur, float thresh,
                    float lowestScale,
                    const DescriptorNormalizerData *normalizer_d,
                    float subsampling, float *memoryTmp, float *memorySub);
void ExtractSiftOctave(const DetectorConfigDevice *cfg, SiftData &siftData,
                       CudaImage &img, int octave, float thresh,
                       float lowestScale,
                       const DescriptorNormalizerData *normalizer_d,
                       float subsampling, float *memoryTmp);
double ScaleDown(DetectorConfigHost *cfg, CudaImage &res, CudaImage &src,
                 float variance);
double ScaleUp(CudaImage &res, CudaImage &src);
double ComputeOrientations(const DetectorConfigDevice *cfg,
                           cudaTextureObject_t texObj, CudaImage &src,
                           SiftData &siftData, int octave);
double ExtractSiftDescriptors(const DetectorConfigDevice *cfg,
                              cudaTextureObject_t texObj, SiftData &siftData,
                              const DescriptorNormalizerData *normalizer_d,
                              float subsampling, int octave);
double RescalePositions(SiftData &siftData, float scale);
double LowPass(DetectorConfigHost *cfg, CudaImage &res, CudaImage &src,
               float scale);
void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel);
double LaplaceMulti(const DetectorConfigDevice *cfg, cudaTextureObject_t texObj,
                    CudaImage &baseImage, CudaImage *results, int octave);
double FindPointsMulti(const DetectorConfigDevice *cfg, CudaImage *sources,
                       SiftData &siftData, float thresh, float edgeLimit,
                       float factor, float lowestScale, float subsampling,
                       int octave);

} // namespace cudasift
#endif
