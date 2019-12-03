#include "cudasift/Covariance.hpp"
#include "cudasift/cudautils.h"

#include <cublas.h>
#include <cublas_v2.h>
#include <cuda.h>

#define safeCuBLAS(err) ::safeCuBLASImpl(err, __FILE__, __LINE__)
#define safeDriverCall(err) ::safeDriverCallImpl(err, __FILE__, __LINE__)
namespace {

static const char *cublasErrorToString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";

  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";

  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";

  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";

  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";

  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";

  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";

  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  default:
    return "CUBLAS_UNKNOWN_ERROR";
  }
  return nullptr;
}

static void safeDriverCallImpl(const CUresult &status, const char *file,
                               int line) {
  const char *res_name, *res_desc;
  if (status == CUDA_SUCCESS)
    return;
  cuGetErrorName(status, &res_name);
  cuGetErrorString(status, &res_desc);
  fprintf(stderr, "cuda driver error @ %s:%d %s (%s)\n", file, line, res_desc,
          res_name);
  exit(-1);
}

void safeCuBLASImpl(cublasStatus_t status, const char *file, const int line) {
  if (status == CUBLAS_STATUS_SUCCESS)
    return;
  fprintf(stderr, "cublas error @ %s:%d %s\n", file, line,
          cublasErrorToString(status));
  exit(-1);
}
} // namespace

namespace cudasift {

CovarianceEstimator::CovarianceEstimator(int device)
    : device(device), total(0) {
  safeCall(cudaSetDevice(device));
#if 0
  safeCall(cudaStreamCreate((cudaStream_t *)&stream));
#else
  safeCall(cudaStreamCreateWithFlags((cudaStream_t *)&stream,
                                     cudaStreamNonBlocking));
#endif
  safeCuBLAS(cublasCreate((cublasHandle_t *)&cublas));

  safeCall(cudaHostAlloc((void **)&mean, DESCRIPTOR_WIDTH * sizeof(double),
                         cudaHostAllocPortable));
  safeCall(cudaHostAlloc((void **)&covariance,
                         DESCRIPTOR_WIDTH * DESCRIPTOR_WIDTH * sizeof(double),
                         cudaHostAllocPortable));
  safeCall(cudaMalloc((void **)&d_descriptors,
                      DESCRIPTOR_WIDTH * BATCH_SIZE * sizeof(float)));
  safeCall(cudaHostAlloc((void **)&h_descriptors,
                         DESCRIPTOR_WIDTH * BATCH_SIZE * sizeof(float),
                         cudaHostAllocPortable));
  safeCall(cudaMalloc((void **)&agg_mean, DESCRIPTOR_WIDTH * sizeof(double)));
  safeCall(cudaMalloc((void **)&agg_covariance,
                      DESCRIPTOR_WIDTH * DESCRIPTOR_WIDTH * sizeof(double)));
  safeCall(cudaMalloc((void **)&ones, BATCH_SIZE * sizeof(float)));
  safeCall(cudaMalloc((void **)&curr_cov,
                      DESCRIPTOR_WIDTH * DESCRIPTOR_WIDTH * sizeof(float)));
  safeCall(cudaMalloc((void **)&curr_mean, DESCRIPTOR_WIDTH * sizeof(float)));
  float one = 1.f;
  uint32_t one_cast = *(uint32_t *)&one;
  safeDriverCall(cuMemsetD32Async((unsigned long int)ones, one_cast, BATCH_SIZE,
                                  (cudaStream_t)stream));
  safeCall(cudaMemsetAsync(agg_mean, 0x00, sizeof(double) * DESCRIPTOR_WIDTH,
                           (cudaStream_t)stream));
  safeCall(cudaMemsetAsync(agg_covariance, 0x00,
                           sizeof(double) * DESCRIPTOR_WIDTH * DESCRIPTOR_WIDTH,
                           (cudaStream_t)stream));
}

void CovarianceEstimator::addDescriptorsDevice(const float *descriptors, int N,
                                               int stride) {
  std::lock_guard lock(mutex);
  processDescriptors(descriptors, stride, N);
}
void CovarianceEstimator::addDescriptors(const float *descriptors, int N,
                                         int stride) {
  std::lock_guard lock(mutex);
  safeCall(cudaSetDevice(device));

  for (int offset = 0; offset < N; offset += BATCH_SIZE) {
    int batch_size = std::min(N - offset, BATCH_SIZE);
    if (stride == DESCRIPTOR_WIDTH) {
      memcpy(h_descriptors, descriptors + offset * DESCRIPTOR_WIDTH,
             batch_size * DESCRIPTOR_WIDTH * sizeof(float));
    } else {
      for (int i = 0; i < batch_size; ++i)
        memcpy(h_descriptors + i * DESCRIPTOR_WIDTH,
               descriptors + (offset + i) * stride,
               sizeof(float) * DESCRIPTOR_WIDTH);
    }
    safeCall(cudaMemcpyAsync(d_descriptors, h_descriptors,
                             batch_size * DESCRIPTOR_WIDTH * sizeof(float),
                             cudaMemcpyHostToDevice, (cudaStream_t)stream));
    processDescriptors(d_descriptors, DESCRIPTOR_WIDTH, batch_size);
  }
}

void CovarianceEstimator::processDescriptors(const float *descriptors,
                                             int stride, int batchSize) {
  safeCuBLAS(cublasSetStream((cublasHandle_t)cublas, (cudaStream_t)stream));
  // compute sum
  float alpha = 1.f, beta = 0.f;
  safeCuBLAS(cublasSgemv((cublasHandle_t)cublas, CUBLAS_OP_N, DESCRIPTOR_WIDTH,
                         batchSize, &alpha, descriptors, stride, ones, 1, &beta,
                         curr_mean, 1));
  // compute cov
  safeCuBLAS(cublasSgemm((cublasHandle_t)cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                         DESCRIPTOR_WIDTH, DESCRIPTOR_WIDTH, batchSize, &alpha,
                         descriptors, stride, descriptors, stride, &beta,
                         curr_cov, DESCRIPTOR_WIDTH));
  // accumulate
  accumulate(batchSize);
}

void CovarianceEstimator::transferToHost() {
  safeCall(cudaMemcpyAsync(covariance, agg_covariance,
                           DESCRIPTOR_WIDTH * DESCRIPTOR_WIDTH * sizeof(double),
                           cudaMemcpyDeviceToHost, (cudaStream_t)stream));
  safeCall(cudaMemcpyAsync(mean, agg_mean, DESCRIPTOR_WIDTH * sizeof(double),
                           cudaMemcpyDeviceToHost, (cudaStream_t)stream));
  safeCall(cudaStreamSynchronize((cudaStream_t)stream));
}

void CovarianceEstimator::getResults(double *user_mean,
                                     double *user_covariance) {
  transferToHost();
  if (user_mean) {
    memcpy(user_mean, mean, DESCRIPTOR_WIDTH * sizeof(double));
  }
  if (user_covariance) {
    memcpy(user_covariance, covariance,
           DESCRIPTOR_WIDTH * DESCRIPTOR_WIDTH * sizeof(double));

    for (int i = 0; i < DESCRIPTOR_WIDTH; ++i)
      for (int j = 0; j < DESCRIPTOR_WIDTH; ++j)
        user_covariance[i * DESCRIPTOR_WIDTH + j] -= mean[i] * mean[j];
  }
}

CovarianceEstimator::~CovarianceEstimator() {
  safeCall(cudaFreeHost(h_descriptors));
  safeCall(cudaFreeHost(mean));
  safeCall(cudaFreeHost(covariance));
  safeCall(cudaFree(d_descriptors));
  safeCall(cudaFree(agg_mean));
  safeCall(cudaFree(agg_covariance));
  safeCall(cudaFree(ones));
  safeCall(cudaFree(curr_mean));
  safeCall(cudaFree(curr_cov));
  safeCall(cudaStreamDestroy((cudaStream_t)stream));
  safeCuBLAS(cublasDestroy((cublasHandle_t)cublas));
}

void CovarianceEstimator::mergeWith(CovarianceEstimator *that) {
  std::lock_guard lock_this(mutex), lock_that(that->mutex);
  safeCall(cudaSetDevice(device));
  safeCall(cudaStreamSynchronize((cudaStream_t)stream));
  safeCall(cudaSetDevice(that->device));
  safeCall(cudaStreamSynchronize((cudaStream_t)that->stream));

  const double total_both = total + that->total;
  const double this_scale = total / total_both,
               that_scale = that->total / total_both;
  // transfer to host
  transferToHost();
  that->transferToHost();
  for (int i = 0; i < DESCRIPTOR_WIDTH; ++i)
    mean[i] = this_scale * mean[i] + that_scale * that->mean[i];
  for (int i = 0; i < DESCRIPTOR_WIDTH * DESCRIPTOR_WIDTH; ++i)
    covariance[i] =
        this_scale * covariance[i] + that_scale * that->covariance[i];

  safeCall(cudaMemsetAsync(that->agg_covariance, 0x0,
                           sizeof(double) * DESCRIPTOR_WIDTH * DESCRIPTOR_WIDTH,
                           (cudaStream_t)that->stream));
  safeCall(cudaMemsetAsync(that->agg_mean, 0x0,
                           sizeof(double) * DESCRIPTOR_WIDTH,
                           (cudaStream_t)that->stream));
  safeCall(cudaSetDevice(device));
  safeCall(cudaMemcpyAsync(agg_covariance, covariance,
                           sizeof(double) * DESCRIPTOR_WIDTH * DESCRIPTOR_WIDTH,
                           cudaMemcpyHostToDevice, (cudaStream_t)stream));
  safeCall(cudaMemcpyAsync(agg_mean, mean, sizeof(double) * DESCRIPTOR_WIDTH,
                           cudaMemcpyHostToDevice, (cudaStream_t)stream));
  total += that->total;
  that->total = 0;
}

} // namespace cudasift
