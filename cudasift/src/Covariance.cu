#include "cudasift/Covariance.hpp"
#include "cudasift/cudaImage.h"
#include "cudasift/cudautils.h"

namespace {
const int BLOCK_WIDTH = cudasift::CovarianceEstimator::BLOCK_WIDTH;
}

namespace cudasift {

__global__ void accCov(const int DESCRIPTOR_WIDTH, float *cov_curr,
                       double *cov_agg, double scale_new, double scale_old) {
  const int id_y = blockIdx.y * BLOCK_WIDTH + threadIdx.y;
  const int id_x = blockIdx.x;
  if (id_y >= DESCRIPTOR_WIDTH || id_x >= DESCRIPTOR_WIDTH)
    return;
  const int id = id_y * DESCRIPTOR_WIDTH + id_x;

  cov_agg[id] = cov_agg[id] * scale_old + double(cov_curr[id]) * scale_new;
}

__global__ void accMean(const int DESCRIPTOR_WIDTH, float *mean_curr,
                        double *mean_agg, double scale_new, double scale_old) {
  const int id = blockIdx.y * BLOCK_WIDTH + threadIdx.y;
  if (id >= DESCRIPTOR_WIDTH)
    return;
  mean_agg[id] = mean_agg[id] * scale_old + double(mean_curr[id]) * scale_new;
}

void CovarianceEstimator::accumulate(int batchSize) {
  const double total_acc = total;
  const double new_total = total_acc + batchSize;
  const double scalar_old = total_acc / new_total;
  const double scalar_new = 1. / new_total;
  total += batchSize;

  dim3 blocks(DESCRIPTOR_WIDTH, iDivUp(DESCRIPTOR_WIDTH, BLOCK_WIDTH)),
      threads(1, BLOCK_WIDTH),
      blocks_mean(1, iDivUp(DESCRIPTOR_WIDTH, BLOCK_WIDTH)),
      threads_mean(1, BLOCK_WIDTH);

  accCov<<<blocks, threads, 0, (cudaStream_t)stream>>>(
      DESCRIPTOR_WIDTH, curr_cov, agg_covariance, scalar_new, scalar_old);
  accMean<<<blocks_mean, threads_mean, 0, (cudaStream_t)stream>>>(
      DESCRIPTOR_WIDTH, curr_mean, agg_mean, scalar_new, scalar_old);

  safeCall(cudaStreamSynchronize((cudaStream_t)stream));
}

} // namespace cudasift
