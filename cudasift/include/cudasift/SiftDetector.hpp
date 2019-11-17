#ifndef SIFTDETECTOR_HPP
#define SIFTDETECTOR_HPP

#include <iostream>
#include <vector>

namespace cudasift {

enum NormalizerOp {
  CopyToOutput,
  ComputeL2,
  CopmuteL1,
  DivideByNorm,
  Clamp,
  Add,
  Mul,
  Sqrt,
  OP_LAST
};

constexpr size_t OpDataSize(const NormalizerOp &op) {
  if (op >= OP_LAST)
    return ~(size_t)0;

  switch (op) {
  case Clamp:
    return 1;
  case Add:
    return 128;
  case Mul:
    return 128 * 128;
  default:
    return 0;
  }
}

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

struct DescriptorNormalizer {
  struct NormalizerStep {
    NormalizerStep(const NormalizerOp &op, std::vector<float> data = {});
    template <typename T>
    NormalizerStep(const NormalizerOp &op, const T &t)
        : NormalizerStep(op, std::vector<float>(t)) {}

    NormalizerOp op() const;
    const std::vector<float> &data() const;

  private:
    NormalizerOp _op;
    std::vector<float> _data;
  };

  DescriptorNormalizer(std::istream &stream);
  DescriptorNormalizer(std::vector<NormalizerStep> steps);
  DescriptorNormalizer();

  DescriptorNormalizerData exportNormalizer() const;

  std::vector<NormalizerStep> steps;
  mutable std::vector<float> data;
  mutable std::vector<int> steps_i;
};

struct SiftParams {
  int nFeatures = 32768;

  int numOctaves = 5;
  float initBlur = 1.f;
  float threshold = 3.0f;

  float lowestScale = 0.f;
  bool scaleUp = false;
  bool subsampling = false;

  DescriptorNormalizer normalizer;
};

struct CudaSift {
  CudaSift(const SiftParams siftParams = SiftParams(), int deviceId = 0,
           void *stream = nullptr);
  ~CudaSift();

private:
  int deviceId;
  void *stream;
};

} // namespace cudasift

#endif
