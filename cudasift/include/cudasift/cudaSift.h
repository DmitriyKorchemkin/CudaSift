#ifndef CUDASIFT_H
#define CUDASIFT_H

#include "cudasift/cudaImage.h"

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
  int numPts;         // Number of available Sift points
  int maxPts;         // Number of allocated Sift points
#ifdef MANAGEDMEM
  SiftPoint *m_data;  // Managed data
#else
  SiftPoint *h_data;  // Host (CPU) data
  SiftPoint *d_data;  // Device (GPU) data
#endif

} SiftData;

typedef struct {
  /*
   * Possible normalizer steps:
   *  0. forward internal buffer to output
   *  1. compute l2 norm and cache it
   *  2. compute l1 norm and cache it
   *  3. divide by cached norm element-wise
   *  4. clamp with alpha * accumulated_norm (consumes a single scalar alpha)
   *  5. add 128-element vector (consumes 128 scalars)
   *  6. compute matrix-vector product with 128x128 matrix (consumes
   * 128*128=16384 scalars
   *  7. divide by square root of absolute value element-wise
   *
   *  // TODO: add special handling for target cases (i.e. take positveness of
   * HoG entries into account)
   *
   *  Vanilla SIFT: 1, 4 (0.2), 1, 3, 0
   *  Vanilla RSIFT: 1, 4 (0.2), 2, 3, 0
   *  ZCA-RSIFT 1, 4 (0.2), 2, 3,  5 (-mean), 6 (ZCA), 1, 3, 0
   *  +RSIFT 1, 4 (0.2) 2, 3, 5 (-mean), 6 (ZCA), 2, 3, 7, 0
   */
  int n_steps;
  int n_data;
  int *normalizer_steps;
  float *data;
} DescriptorNormalizerData;

void InitCuda(int devNum = 0);
float *AllocSiftTempMemory(int width, int height, int numOctaves, bool scaleUp = false);
void FreeSiftTempMemory(float *memoryTmp);
void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves,
                 double initBlur, float thresh,
                 const DescriptorNormalizerData &normalizer,
                 float lowestScale = 0.0f, bool scaleUp = false,
                 float *tempMemory = 0);
void InitSiftData(SiftData &data, int num = 1024, bool host = false, bool dev = true);
void FreeSiftData(SiftData &data);
void PrintSiftData(SiftData &data);
double MatchSiftData(SiftData &data1, SiftData &data2);
double FindHomography(SiftData &data,  float *homography, int *numMatches, int numLoops = 1000, float minScore = 0.85f, float maxAmbiguity = 0.95f, float thresh = 5.0f);

#endif
