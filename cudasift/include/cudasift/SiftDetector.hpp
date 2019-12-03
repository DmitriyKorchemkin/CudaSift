#ifndef SIFTDETECTOR_HPP
#define SIFTDETECTOR_HPP

#include "cudasift/Covariance.hpp"
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

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

enum NormalizerOp {
  CopyToOutput,
  ComputeL2,
  ComputeL1,
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
  const int *normalizer_steps;
  const float *data;
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
  void exportNormalizer(std::ostream &ostream, int nStepsOverride = -1) const;

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

// POD wrapper for easier interoperability with whatever library
template <typename T> struct ImgWrapper {
  // pointer to host data
  T *data;
  // width in px
  int width;
  // height in px
  int height;
  // stride in bytes (row-major)
  int stride_bytes;
  // true if image is interleaved bgr
  bool is_bgr;

  // stride < width => stride will be guessed from width (assuming
  ImgWrapper(T *data, int width, int height, bool is_bgr = false,
             int stride_bytes = 0)
      : data(data), width(width), height(height), is_bgr(is_bgr),
        stride_bytes(stride_bytes) {
    if (stride_bytes < width * sizeof(T)) {
      this->stride_bytes = (is_bgr ? 3 * width : width) * sizeof(T);
    }
  }
  size_t total() const { return stride_bytes * height; }
};
using ImageU8 = ImgWrapper<uint8_t>;
using ImageFP32 = ImgWrapper<float>;

// Either mmaped or pinned
struct JPEGImage {
  JPEGImage(const char *filename, bool copyToPinned = false);
  ~JPEGImage();
  bool isPinned() const;
  const uint8_t *data() const;
  int width() const;
  int height() const;
  int size() const;
  const std::string &name() const;

private:
  const std::string file;
  int imgWidth, imgHeight, dataSize;
  int fd;
  uint8_t *data_ptr;
  bool is_pinned;
};

struct SiftDetectorImpl;
struct SiftData;

struct CudaSift {
  // empty deviceIds = allocate on all available devices
  CudaSift(const SiftParams &siftParams = SiftParams(),
           std::vector<int> deviceIds = {}, int streamPerDevice = 2,
           bool collectCovariance = false, bool useNvJpeg = false);
  void detectAndExtract(const ImageU8 &image,
                        std::vector<SiftPoint> &siftPoints);
  void detectAndExtract(const ImageFP32 &image,
                        std::vector<SiftPoint> &siftPoints);
  void detectAndExtract(const JPEGImage &image,
                        std::vector<SiftPoint> &siftPoints);

  void getResults(double *mean, double *covariance);

  ~CudaSift();

private:
  struct DetectorContext {
    DetectorContext(const SiftParams &params, int deviceId, void *stream,
                    bool collectCovariance, int width_init = 4096,
                    int height_init = 4096);
    ~DetectorContext();

    void realloc(int w, int h, int nFeatures, bool needRGB);
    void processBgrU8(int w, int h, int stride_bytes);
    void processU8(int w, int h, int stride_bytes);
    void processFP32(int w, int h, int stride_bytes);
    void processJPEG(const uint8_t *data, int w, int h, int size_bytes);

    std::unique_ptr<SiftDetectorImpl> detector;

    int alloc_u8 = 0, alloc_float = 0, alloc_features = 0;
    uint8_t *u8_pinned = nullptr, *u8_device = nullptr;
    float *float_device = nullptr, *float_pinned = nullptr;
    std::unique_ptr<SiftData> siftData;
    void *stream;
    void *nvjpegState;
    void *tjState;
    int device;
    void freeMemory();
    std::unique_ptr<CovarianceEstimator> covariance;
    bool collectCovariance;

  private:
    DetectorContext(const DetectorContext &ctx) = delete;
  };
  std::vector<std::unique_ptr<DetectorContext>> detectors;
  std::queue<DetectorContext *> freeDetectors;
  std::mutex mutex;
  std::condition_variable detectorAvailable;
  bool collectCovariance, useNvJpeg;
};

} // namespace cudasift

#endif
