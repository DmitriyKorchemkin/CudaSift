#ifndef COVARIANCE
#define COVARIANCE

#include <mutex>

namespace cudasift {

struct CovarianceEstimator {
  CovarianceEstimator(int device);
  ~CovarianceEstimator();
  void addDescriptors(const float *descriptors, int N, int stride);
  void addDescriptorsDevice(const float *descriptors, int N, int stride);

  void getResults(double *mean, double *covariance);
  static constexpr int BATCH_SIZE = 65536;
  static constexpr int DESCRIPTOR_WIDTH = 128;
  static constexpr int BLOCK_WIDTH = 32;

  // combines mean / covariance estimates in this estimator
  // and clears state of other estimator
  void mergeWith(CovarianceEstimator *other);

private:
  void transferToHost();
  void processDescriptors(const float *descriptors, int stride, int batchSize);
  void accumulate(int batchSize);

  float *d_descriptors, *h_descriptors, *ones;
  float *curr_mean, *curr_cov;
  double *agg_mean, *agg_covariance;
  double *mean, *covariance;
  std::mutex mutex;
  int device;
  size_t total;
  void *stream, *cublas;
};

} // namespace cudasift

#endif
