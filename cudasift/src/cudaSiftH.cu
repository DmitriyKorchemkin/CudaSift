//********************************************************//
// CUDA SIFT extractor by Mårten Björkman aka Celebrandil //
//********************************************************//

#include "cudasift/cudautils.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "cudasift/cudaImage.h"
#include "cudasift/cudaSift.h"
#include "cudasift/cudaSiftD.h"
#include "cudasift/cudaSiftH.h"

#include "cudaSiftD.cu"

namespace cudasift {

void InitCuda(int devNum) {
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  if (!nDevices) {
    std::cerr << "No CUDA devices available" << std::endl;
    return;
  }
  devNum = std::min(nDevices - 1, devNum);
  deviceInit(devNum);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, devNum);
  printf("Device Number: %d\n", devNum);
  printf("  Device name: %s\n", prop.name);
  printf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate / 1000);
  printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
  printf("  Peak Memory Bandwidth (GB/s): %.1f\n\n",
         2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
}

SiftData::~SiftData() {
  safeCall(cudaSetDevice(deviceId));
#ifdef MANAGEDMEM
  safeCall(cudaFree(m_data));
#else
  if (d_data != NULL)
    safeCall(cudaFree(d_data));
  d_data = NULL;
  if (h_data != NULL)
    safeCall(cudaFreeHost(h_data));
#endif
  numPts = 0;
  maxPts = 0;
}

SiftData::SiftData(SiftData &&other) {
  std::swap(maxPts, other.maxPts);
#ifdef MANAGEDMEM
  std::swap(m_data, other.m_data);
#else
  std::swap(h_data, other.h_data);
  std::swap(d_data, other.d_data);
#endif
}

SiftData::SiftData(const int maxPts) : maxPts(maxPts) {}

void FreeSiftTempMemory(float *memoryTmp) {
  if (memoryTmp)
    safeCall(cudaFree(memoryTmp));
}

SiftDetectorImpl::SiftDetectorImpl(const SiftParams &params, int device,
                                   cudaStream_t stream)
    : configHost(params.nFeatures, stream), device(device),
      stream((cudaStream_t)stream), params(params) {
  cudaSetDevice(device);
  float kernel[12 * 8 * 16];
  PrepareLaplaceKernels(params.numOctaves, 0.f, kernel);

  safeCall(cudaMemcpyAsync(configHost.laplaceKernel, kernel,
                           8 * 12 * 16 * sizeof(float), cudaMemcpyHostToDevice,
                           stream));

  const auto &normalizer_h = params.normalizer.exportNormalizer();
  int sz = sizeof(DescriptorNormalizerData) +
           normalizer_h.n_steps * sizeof(int) +
           normalizer_h.n_data * sizeof(float);
  DescriptorNormalizerData normalizer_d;
  normalizer_d.n_steps = normalizer_h.n_steps;
  normalizer_d.n_data = normalizer_h.n_data;
  safeCall(cudaMalloc((void **)&p_normalizer_d, sz));
  normalizer_d.normalizer_steps = (int *)(void *)(p_normalizer_d + 1);
  normalizer_d.data =
      ((float *)((int *)(void *)(p_normalizer_d + 1) + normalizer_d.n_steps));
  safeCall(cudaMemcpyAsync(p_normalizer_d, &normalizer_d,
                           sizeof(DescriptorNormalizerData),
                           cudaMemcpyHostToDevice, stream));
  safeCall(cudaMemcpyAsync(const_cast<int *>(normalizer_d.normalizer_steps),
                           normalizer_h.normalizer_steps,
                           sizeof(int) * normalizer_h.n_steps,
                           cudaMemcpyHostToDevice, stream));
  safeCall(cudaMemcpyAsync(
      const_cast<float *>(normalizer_d.data), normalizer_h.data,
      sizeof(float) * normalizer_h.n_data, cudaMemcpyHostToDevice, stream));
  cudaStreamSynchronize(stream);
  checkMsg("Normalizer allocation failed\n");
}

SiftDetectorImpl::~SiftDetectorImpl() {
  FreeSiftTempMemory(memoryTmp);
  safeCall(cudaFree(p_normalizer_d));
}

SiftPoint *SiftData::hostPtr() {
#ifdef MANAGEDMEM
  return m_data;
#else
  return h_data;
#endif
}

void SiftData::transferAsync(bool toHost, void *stream) {
#ifndef MANAGEDMEM
  if (toHost) {
    safeCall(cudaMemcpyAsync(h_data, d_data, numPts * sizeof(SiftPoint),
                             cudaMemcpyDeviceToHost, (cudaStream_t)stream));
  } else {
    safeCall(cudaMemcpyAsync(d_data, h_data, numPts * sizeof(SiftPoint),
                             cudaMemcpyHostToDevice, (cudaStream_t)stream));
  }
#endif
}

void SiftData::transfer(bool toHost, void *stream, bool sync) {
#ifndef MANAGEDMEM
  if (toHost) {
    safeCall(cudaMemcpyAsync(h_data, d_data, numPts * sizeof(SiftPoint),
                             cudaMemcpyDeviceToHost, (cudaStream_t)stream));
  } else {
    safeCall(cudaMemcpyAsync(d_data, h_data, numPts * sizeof(SiftPoint),
                             cudaMemcpyHostToDevice, (cudaStream_t)stream));
  }
#endif
  if (sync) {
    safeCall(cudaStreamSynchronize((cudaStream_t)stream));
  }
}

void SiftDetectorImpl::ExtractSift(SiftData &siftData, CudaImage &img,
                                   bool syncOnTransfer) {
  cudaSetDevice(device);
  TimerGPU timer(stream);
  const int nd = NUM_SCALES + 3;
  int w = img.width * (params.scaleUp ? 2 : 1);
  int h = img.height * (params.scaleUp ? 2 : 1);
  int p = iAlignUp(w, 128);
  int width = w, height = h;
  int size = h * p;         // image sizes
  int sizeTmp = nd * h * p; // laplace buffer sizes
  for (int i = 0; i < params.numOctaves; i++) {
    w /= 2;
    h /= 2;
    int p = iAlignUp(w, 128);
    size += h * p;
    sizeTmp += nd * h * p;
  }
  size += sizeTmp;
  if (!memoryTmp || memoryAlloc < size) {

    FreeSiftTempMemory(memoryTmp);
    size_t pitch;
    safeCall(cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096,
                             (size + 4095) / 4096 * sizeof(float)));
    memoryAlloc = size;

#ifdef VERBOSE
    printf("Allocated memory size: %d bytes\n", size);
    printf("Memory allocation time =      %.2f ms\n\n", timer.read());
#endif
  }

  // memoryAlloc << std::endl;
  float *memorySub = memoryTmp + sizeTmp;
  safeCall(cudaMemsetAsync(configHost.pointCounter, 0x0,
                           sizeof(uint32_t) * (8 * 2 + 1), stream));
  lowImg.Allocate(width, height, iAlignUp(width, 128), false, memorySub);

  if (!params.scaleUp) {
    LowPass(lowImg, img, max(params.initBlur, 0.001f));
    TimerGPU timer1(stream);
    ExtractSiftLoop(siftData, lowImg, params.numOctaves, 0.0f,
                    params.lowestScale, 1.0f,
                    memorySub + height * iAlignUp(width, 128));
    safeCall(cudaMemcpyAsync(&siftData.numPts,
                             &configHost.pointCounter[2 * params.numOctaves],
                             sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    safeCall(cudaStreamSynchronize(stream));
    siftData.numPts =
        (siftData.numPts < siftData.maxPts ? siftData.numPts : siftData.maxPts);
#ifdef VERBOSE
    printf("SIFT extraction time =        %.2f ms %d\n", timer1.read(),
           siftData.numPts);
#endif
  } else {
    CudaImage upImg;
    upImg.Allocate(width, height, iAlignUp(width, 128), false, memoryTmp);
    TimerGPU timer1(stream);
    ScaleUp(upImg, img);
    LowPass(lowImg, upImg, max(params.initBlur, 0.001f));
    ExtractSiftLoop(siftData, lowImg, params.numOctaves, 0.0f,
                    params.lowestScale * 2.0f, 1.0f,
                    memorySub + height * iAlignUp(width, 128));
    safeCall(cudaMemcpyAsync(&siftData.numPts,
                             &configHost.pointCounter[2 * params.numOctaves],
                             sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    safeCall(cudaStreamSynchronize(stream));
    siftData.numPts =
        (siftData.numPts < siftData.maxPts ? siftData.numPts : siftData.maxPts);
    RescalePositions(siftData, 0.5f);
#ifdef VERBOSE
    printf("SIFT extraction time =        %.2f ms\n", timer1.read());
#endif
  }

  siftData.transfer(true, stream, syncOnTransfer);
  double totTime = timer.read();
#ifdef VERBOSE
  printf("Incl prefiltering & memcpy =  %.2f ms %d\n\n", totTime,
         siftData.numPts);
#endif
}

int SiftDetectorImpl::ExtractSiftLoop(SiftData &siftData, CudaImage &img,
                                      int numOctaves, float blur,
                                      float lowestScale, float subsampling,
                                      float *memorySub) {
#ifdef VERBOSE
  TimerGPU timer(stream);
#endif
  int w = img.width;
  int h = img.height;
  if (numOctaves > 1) {
    CudaImage subImg;
    int p = iAlignUp(w / 2, 128);
    subImg.Allocate(w / 2, h / 2, p, false, memorySub);
    ScaleDown(subImg, img, 0.5f);
    float totInitBlur = (float)sqrt(blur * blur + 0.5f * 0.5f) / 2.0f;
    ExtractSiftLoop(siftData, subImg, numOctaves - 1, totInitBlur, lowestScale,
                    subsampling * 2.0f, memorySub + (h / 2) * p);
  }
  ExtractSiftOctave(siftData, img, numOctaves, lowestScale, subsampling);
#ifdef VERBOSE
  double totTime = timer.read();
  printf("ExtractSift time total =      %.2f ms %d\n\n", totTime, numOctaves);
#endif
  return 0;
}

void SiftDetectorImpl::ExtractSiftOctave(SiftData &siftData, CudaImage &img,
                                         int octave, float lowestScale,
                                         float subsampling) {
  const int nd = NUM_SCALES + 3;
#ifdef VERBOSE
  safeCall(cudaStreamSynchronize(stream));
  unsigned int fstPts, totPts;
  fstPts = configHost->pointCounter[2 * octave - 1];
  TimerGPU timer0;
#endif
  CudaImage diffImg[nd];
  int w = img.width;
  int h = img.height;
  int p = iAlignUp(w, 128);
  for (int i = 0; i < nd - 1; i++) {
    diffImg[i].Allocate(w, h, p, false, memoryTmp + i * p * h);

    //    memoryTmp<< "..." << memoryTmp+(i+1)*p*h-memoryTmp << std::endl;
  }

  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = img.d_data;
  resDesc.res.pitch2D.width = img.width;
  resDesc.res.pitch2D.height = img.height;
  resDesc.res.pitch2D.pitchInBytes = img.pitch * sizeof(float);
  resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;
  // Create texture object
  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

#ifdef VERBOSE
  TimerGPU timer1;
#endif
  float baseBlur = pow(2.0f, -1.0f / NUM_SCALES);
  float diffScale = pow(2.0f, 1.0f / NUM_SCALES);

  LaplaceMulti(texObj, img, diffImg, octave);

  FindPointsMulti(diffImg, siftData, 10.0f, 1.0f / NUM_SCALES,
                  lowestScale / subsampling, subsampling, octave);
#ifdef VERBOSE
  double gpuTimeDoG = timer1.read();
  TimerGPU timer4;
#endif

  ComputeOrientations(texObj, img, siftData, octave);

  ExtractSiftDescriptors(texObj, siftData, subsampling, octave);

  ;
  safeCall(cudaDestroyTextureObject(texObj));
#ifdef VERBOSE
  double gpuTimeSift = timer4.read();
  double totTime = timer0.read();
  printf("GPU time : %.2f ms + %.2f ms + %.2f ms = %.2f ms\n",
         totTime - gpuTimeDoG - gpuTimeSift, gpuTimeDoG, gpuTimeSift, totTime);
  safeCall(cudaMemcpyAsync(&totPts, &d_PointCounterAddr[2 * octave + 1],
                           sizeof(int), cudaMemcpyDeviceToHost, stream));
  totPts = (totPts < siftData.maxPts ? totPts : siftData.maxPts);
  if (totPts > 0)
    printf("           %.2f ms / DoG,  %.4f ms / Sift,  #Sift = %d\n",
           gpuTimeDoG / NUM_SCALES, gpuTimeSift / (totPts - fstPts),
           totPts - fstPts);
#endif
}

void SiftData::allocate(int num, bool host, bool dev, int device) {
  numPts = 0;
  maxPts = num;
  int sz = sizeof(SiftPoint) * num;
  cudaSetDevice(device);
  deviceId = device;
#ifdef MANAGEDMEM
  safeCall(cudaMallocManaged((void **)&m_data, sz));
#else
  h_data = NULL;
  if (host)
    safeCall(cudaHostAlloc((void **)&h_data, sz, cudaHostAllocPortable));
  d_data = NULL;
  if (dev) {
    safeCall(cudaMalloc((void **)&d_data, sz));
  }
#endif
}

void PrintSiftData(SiftData &data) {
#ifdef MANAGEDMEM
  SiftPoint *h_data = data.m_data;
#else
  SiftPoint *h_data = data.h_data;
  if (data.h_data == NULL) {
    h_data = (SiftPoint *)malloc(sizeof(SiftPoint) * data.maxPts);
    safeCall(cudaMemcpy(h_data, data.d_data, sizeof(SiftPoint) * data.numPts,
                        cudaMemcpyDeviceToHost));
    data.h_data = h_data;
  }
#endif
  for (int i = 0; i < data.numPts; i++) {
    printf("xpos         = %.2f\n", h_data[i].xpos);
    printf("ypos         = %.2f\n", h_data[i].ypos);
    printf("scale        = %.2f\n", h_data[i].scale);
    printf("sharpness    = %.2f\n", h_data[i].sharpness);
    printf("edgeness     = %.2f\n", h_data[i].edgeness);
    printf("orientation  = %.2f\n", h_data[i].orientation);
    printf("score        = %.2f\n", h_data[i].score);
    float *siftData = (float *)&h_data[i].data;
    for (int j = 0; j < 8; j++) {
      if (j == 0)
        printf("data = ");
      else
        printf("       ");
      for (int k = 0; k < 16; k++)
        if (siftData[j + 8 * k] < 0.05)
          printf(" .   ");
        else
          printf("%.2f ", siftData[j + 8 * k]);
      printf("\n");
    }
  }
  printf("Number of available points: %d\n", data.numPts);
  printf("Number of allocated points: %d\n", data.maxPts);
}

///////////////////////////////////////////////////////////////////////////////
// Host side master functions
///////////////////////////////////////////////////////////////////////////////

double SiftDetectorImpl::ScaleDown(CudaImage &res, CudaImage &src,
                                   float variance) {
  if (res.d_data == NULL || src.d_data == NULL) {
    printf("ScaleDown: missing data\n");
    return 0.0;
  }
  if (configHost.oldVarianceScaleDown != variance) {
    float h_Kernel[5];
    float kernelSum = 0.0f;
    for (int j = 0; j < 5; j++) {
      h_Kernel[j] = (float)expf(-(double)(j - 2) * (j - 2) / 2.0 / variance);
      kernelSum += h_Kernel[j];
    }
    for (int j = 0; j < 5; j++)
      h_Kernel[j] /= kernelSum;
    configHost.oldVarianceScaleDown = variance;
    safeCall(cudaMemcpyAsync(configHost.scaleDownKernel, h_Kernel,
                             5 * sizeof(float), cudaMemcpyHostToDevice,
                             stream));
  }
  dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
  dim3 threads(SCALEDOWN_W + 4);
  cudasift::ScaleDown<<<blocks, threads, 0, stream>>>(
      configHost.dev, res.d_data, src.d_data, src.width, src.pitch, src.height,
      res.pitch);

  cudaStreamSynchronize(stream);
  checkMsg("ScaleDown() execution failed\n");
  return 0.0;
}

double SiftDetectorImpl::ScaleUp(CudaImage &res, CudaImage &src) {
  if (res.d_data == NULL || src.d_data == NULL) {
    printf("ScaleUp: missing data\n");
    return 0.0;
  }
  dim3 blocks(iDivUp(res.width, SCALEUP_W), iDivUp(res.height, SCALEUP_H));
  dim3 threads(SCALEUP_W / 2, SCALEUP_H / 2);
  cudasift::ScaleUp<<<blocks, threads, 0, stream>>>(
      res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
  cudaStreamSynchronize(stream);
  checkMsg("ScaleUp() execution failed\n");
  return 0.0;
}

double SiftDetectorImpl::ComputeOrientations(cudaTextureObject_t texObj,
                                             CudaImage &src, SiftData &siftData,
                                             int octave) {
  dim3 blocks(512);
#ifdef MANAGEDMEM
  ComputeOrientationsCONST<<<blocks, threads, 0, stream>>>(
      configHost.dev, texObj, siftData.m_data, octave);
#else
  dim3 threads(11 * 11);
  ComputeOrientationsCONST<<<blocks, threads, 0, stream>>>(
      configHost.dev, texObj, siftData.d_data, octave);
#endif
  cudaStreamSynchronize(stream);
  checkMsg("ComputeOrientations() execution failed\n");
  return 0.0;
}

double SiftDetectorImpl::ExtractSiftDescriptors(cudaTextureObject_t texObj,
                                                SiftData &siftData,
                                                float subsampling, int octave) {
  dim3 blocks(512);
  dim3 threads(16, 8);
#ifdef MANAGEDMEM
  ExtractSiftDescriptorsCONST<<<blocks, threads, 0, stream>>>(
      configHost.dev, texObj, siftData.m_data, p_normalizer_d, subsampling,
      octave);
#else
  ExtractSiftDescriptorsCONSTNew<<<blocks, threads, 0, stream>>>(
      configHost.dev, texObj, siftData.d_data, p_normalizer_d, subsampling,
      octave);
#endif
  cudaStreamSynchronize(stream);
  checkMsg("ExtractSiftDescriptors() execution failed\n");
  return 0.0;
}

double SiftDetectorImpl::RescalePositions(SiftData &siftData, float scale) {
  dim3 blocks(iDivUp(siftData.numPts, 64));
  dim3 threads(64);
  cudasift::RescalePositions<<<blocks, threads, 0, stream>>>(
      siftData.d_data, siftData.numPts, scale);
  cudaStreamSynchronize(stream);
  checkMsg("RescapePositions() execution failed\n");
  return 0.0;
}

double SiftDetectorImpl::LowPass(CudaImage &res, CudaImage &src, float scale) {
  if (scale != configHost.oldScaleLowPass) {
    float kernel[2 * LOWPASS_R + 1];
    float kernelSum = 0.0f;
    float ivar2 = 1.0f / (2.0f * scale * scale);
    for (int j = -LOWPASS_R; j <= LOWPASS_R; j++) {
      kernel[j + LOWPASS_R] = (float)expf(-(double)j * j * ivar2);
      kernelSum += kernel[j + LOWPASS_R];
    }
    for (int j = -LOWPASS_R; j <= LOWPASS_R; j++)
      kernel[j + LOWPASS_R] /= kernelSum;
    safeCall(cudaMemcpyAsync(configHost.lowPassKernel, kernel,
                             (2 * LOWPASS_R + 1) * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
    configHost.oldScaleLowPass = scale;
  }
  int width = res.width;
  int pitch_res = res.pitch;
  int pitch_src = src.pitch;
  int height = res.height;
  dim3 blocks(iDivUp(width, LOWPASS_W), iDivUp(height, LOWPASS_H));

  dim3 threads(LOWPASS_W + 2 * LOWPASS_R, 4);
  LowPassBlock<<<blocks, threads, 0, stream>>>(configHost.dev, src.d_data,
                                               res.d_data, width, pitch_src,
                                               pitch_res, height);
  cudaStreamSynchronize(stream);
  checkMsg("LowPass() execution failed\n");
  return 0.0;
}

//==================== Multi-scale functions ===================//

void SiftDetectorImpl::PrepareLaplaceKernels(int numOctaves, float initBlur,
                                             float *kernel) {
  if (numOctaves > 1) {
    float totInitBlur = (float)sqrt(initBlur * initBlur + 0.5f * 0.5f) / 2.0f;
    PrepareLaplaceKernels(numOctaves - 1, totInitBlur, kernel);
  }
  float scale = pow(2.0f, -1.0f / NUM_SCALES);
  float diffScale = pow(2.0f, 1.0f / NUM_SCALES);
  for (int i = 0; i < NUM_SCALES + 3; i++) {
    float kernelSum = 0.0f;
    float var = scale * scale - initBlur * initBlur;
    for (int j = 0; j <= LAPLACE_R; j++) {
      kernel[numOctaves * 12 * 16 + 16 * i + j] =
          (float)expf(-(double)j * j / 2.0 / var);
      kernelSum += (j == 0 ? 1 : 2) * kernel[numOctaves * 12 * 16 + 16 * i + j];
    }
    for (int j = 0; j <= LAPLACE_R; j++)
      kernel[numOctaves * 12 * 16 + 16 * i + j] /= kernelSum;
    scale *= diffScale;
  }
}

double SiftDetectorImpl::LaplaceMulti(cudaTextureObject_t texObj,
                                      CudaImage &baseImage, CudaImage *results,
                                      int octave) {
  int width = results[0].width;
  int pitch = results[0].pitch;
  int height = results[0].height;

  dim3 threads(LAPLACE_W + 2 * LAPLACE_R);
  dim3 blocks(iDivUp(width, LAPLACE_W), height);
  LaplaceMultiMem<<<blocks, threads, 0, stream>>>(
      configHost.dev, baseImage.d_data, results[0].d_data, width, pitch, height,
      octave);
  cudaStreamSynchronize(stream);
  checkMsg("LaplaceMulti() execution failed\n");
  return 0.0;
}

double SiftDetectorImpl::FindPointsMulti(CudaImage *sources, SiftData &siftData,
                                         float edgeLimit, float factor,
                                         float lowestScale, float subsampling,
                                         int octave) {
  if (sources->d_data == NULL) {
    printf("FindPointsMulti: missing data\n");
    return 0.0;
  }
  int w = sources->width;
  int p = sources->pitch;
  int h = sources->height;

  dim3 blocks(iDivUp(w, MINMAX_W) * NUM_SCALES, iDivUp(h, MINMAX_H));
  dim3 threads(MINMAX_W + 2);
#ifdef MANAGEDMEM
  FindPointsMulti<<<blocks, threads, 0, stream>>>(
      configHost.dev, sources->d_data, siftData.m_data, w, p, h, subsampling,
      lowestScale, params.threshold, factor, edgeLimit, octave);
#else

  FindPointsMultiNew<<<blocks, threads, 0, stream>>>(
      configHost.dev, sources->d_data, siftData.d_data, w, p, h, subsampling,
      lowestScale, params.threshold, factor, edgeLimit, octave);
#endif
  cudaStreamSynchronize(stream);
  checkMsg("FindPointsMulti() execution failed\n");
  return 0.0;
}

DetectorConfigHost::DetectorConfigHost(int nPoints, cudaStream_t stream) {
  safeCall(cudaMalloc((void **)&dev, sizeof(DetectorConfigDevice)));
  oldScaleLowPass = oldVarianceScaleDown = -1.f;
  void *ptrs[3];
  new (dev) DetectorConfigDevice(nPoints, ptrs, stream);

  scaleDownKernel = (float *)ptrs[0];
  lowPassKernel = (float *)ptrs[1];
  laplaceKernel = (float *)ptrs[2];
  pointCounter = &dev->pointCounter[0];
}

DetectorConfigHost::~DetectorConfigHost() {
  dev->~DetectorConfigDevice();
  safeCall(cudaFree(dev));
  safeCall(cudaFree(scaleDownKernel));
  safeCall(cudaFree(lowPassKernel));
  safeCall(cudaFree(laplaceKernel));
}

DetectorConfigDevice::DetectorConfigDevice(int maxNumPoints, void **ptrs,
                                           cudaStream_t stream) {
  cudaMemsetAsync(pointCounter, 0x0, sizeof(uint32_t) * (8 * 2 + 1), stream);

  safeCall(cudaMalloc((void **)&ptrs[0], sizeof(float) * 5));
  safeCall(cudaMalloc((void **)&ptrs[1], sizeof(float) * (2 * LOWPASS_R + 1)));
  safeCall(cudaMalloc((void **)&ptrs[2], sizeof(float) * (8 * 12 * 16)));
  safeCall(cudaMemcpyAsync(&scaleDownKernel, ptrs, sizeof(float *) * 3,
                           cudaMemcpyHostToDevice, stream));
  safeCall(cudaMemcpyAsync(&this->maxNumPoints, &maxNumPoints, sizeof(int),
                           cudaMemcpyHostToDevice, stream));
}

DetectorConfigDevice::~DetectorConfigDevice() {}

} // namespace cudasift
