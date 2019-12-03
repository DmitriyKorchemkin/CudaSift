#include "cudasift/SiftDetector.hpp"
#include "cudasift/cudaSift.h"
#include "cudasift/cudautils.h"
#include "cudasift/nvjpegUtils.h"

#include <cstring>
#include <limits>
#include <type_traits>

#include <fcntl.h>
#include <filesystem>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <turbojpeg.h>

namespace fs = std::filesystem;

namespace cudasift {
bool JPEGImage::isPinned() const { return is_pinned; }

const uint8_t *JPEGImage::data() const { return data_ptr; }

int JPEGImage::width() const { return imgWidth; }

int JPEGImage::height() const { return imgHeight; }

int JPEGImage::size() const { return dataSize; }

const std::string &JPEGImage::name() const { return file; }

JPEGImage::JPEGImage(const char *filename, bool copyToPinned)
    : is_pinned(copyToPinned), file(filename) {
  fs::path p(filename);
  dataSize = fs::file_size(p);

  fd = open(filename, O_RDONLY);
  void *mapped =
      mmap(nullptr, dataSize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
  madvise(mapped, dataSize, MADV_WILLNEED | MADV_SEQUENTIAL);

  if (is_pinned) {
    // need to allocate pinned memory & memcpy
    safeCall(
        cudaHostAlloc((void **)&data_ptr, dataSize, cudaHostAllocPortable));
    memcpy(data_ptr, mapped, dataSize);
    munmap(mapped, dataSize);
    close(fd);
  } else {
    // leave mmaped memory as-is
    data_ptr = (uint8_t *)mapped;
  }

  if (dataSize < 12) {
    throw std::runtime_error("Insufficient file size: " + file);
  }
  int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT], nComponents;
  nvjpegChromaSubsampling_t subsampling;
  nvjpegGetImageInfo(NVJPEG::GetHandle(), data_ptr, dataSize, &nComponents,
                     &subsampling, widths, heights);
  imgWidth = widths[0];
  imgHeight = heights[0];
}

JPEGImage::~JPEGImage() {
  if (is_pinned) {
    safeCall(cudaFreeHost(data_ptr));
  } else {
    munmap(data_ptr, dataSize);
    close(fd);
  }
}

DescriptorNormalizer::NormalizerStep::NormalizerStep(const NormalizerOp &op,
                                                     std::vector<float> data)
    : _op(op), _data(std::move(data)) {
  if (OpDataSize(_op) != _data.size())
    throw std::runtime_error(
        "Invalid payload size (" + std::to_string(_data.size()) +
        ", expected: " + std::to_string(OpDataSize(_op)) + ")");
}

NormalizerOp DescriptorNormalizer::NormalizerStep::op() const { return _op; }

const std::vector<float> &DescriptorNormalizer::NormalizerStep::data() const {
  return _data;
}

DescriptorNormalizer::DescriptorNormalizer(std::istream &stream) {
  uint32_t n_steps;
  std::cout << "Reading normalizer config: ";
  stream >> n_steps;
  std::cout << n_steps << " steps\n";

  for (int i = 0; i < n_steps; ++i) {
    std::underlying_type<NormalizerOp>::type op_id;
    stream >> op_id;
    auto op = static_cast<NormalizerOp>(op_id);
    auto size = OpDataSize(op);
    if (size == std::numeric_limits<size_t>::max())
      throw std::runtime_error("Invalid normalizer operation: " +
                               std::to_string(op_id));

    std::vector<float> vec(size);
    for (int j = 0; j < size; ++j)
      stream >> vec[j];
    std::cout << "\t" << op_id << ", " << size << " data-fields: [";
    for (auto &d : vec)
      std::cout << d << " ";
    std::cout << "]\n";

    steps.emplace_back(op, std::move(vec));
  }
}

void DescriptorNormalizer::exportNormalizer(std::ostream &os,
                                            int nsteps) const {
  if (nsteps == -1)
    nsteps = steps.size();
  os << nsteps << '\n';
  for (auto &s : steps) {
    os << s.op();
    switch (s.data().size()) {
    case 1:
      os << ' ' << s.data()[0];
    case 0:
      os << '\n';
      continue;
    default:
      os << '\n';
    }
    for (auto &d : s.data())
      os << std::setprecision(16) << std::showpos << std::scientific << d
         << std::noshowpos << (&d == &*s.data().rbegin() ? '\n' : ' ');
  }
}

DescriptorNormalizer::DescriptorNormalizer(std::vector<NormalizerStep> steps)
    : steps(std::move(steps)) {}

DescriptorNormalizer::DescriptorNormalizer() {
  steps.emplace_back(ComputeL2);
  steps.emplace_back(Clamp, std::vector<float>({0.2f}));
  steps.emplace_back(ComputeL2);
  steps.emplace_back(DivideByNorm);
  steps.emplace_back(CopyToOutput);
}

DescriptorNormalizerData DescriptorNormalizer::exportNormalizer() const {
  size_t total = 0;
  data.clear();
  steps_i.clear();

  for (auto &step : steps) {
    const auto &payload = step.data();
    total += payload.size();
    data.insert(data.end(), payload.begin(), payload.end());
    steps_i.push_back(static_cast<int>(step.op()));
  }

  DescriptorNormalizerData ndata;
  ndata.n_data = total;
  ndata.n_steps = steps.size();
  ndata.normalizer_steps = steps_i.data();
  ndata.data = data.data();

  return ndata;
}

CudaSift::CudaSift(const SiftParams &siftParams, std::vector<int> deviceIds,
                   int streamPerDevice, bool collectCovariance, bool useNvJpeg)
    : collectCovariance(collectCovariance), useNvJpeg(useNvJpeg) {
  // std::cout << "Device ids-in: " << deviceIds.size() << std::endl;
  if (!deviceIds.size()) {
    int devices;
    safeCall(cudaGetDeviceCount(&devices));
    for (int i = 0; i < devices; ++i)
      deviceIds.push_back(i);
  }
  //    std::cout << "Device ids-out " << deviceIds.size() << std::endl;
  for (auto &id : deviceIds) {
    for (int j = 0; j < streamPerDevice; ++j) {
      safeCall(cudaSetDevice(id));
      cudaStream_t stream;
#if 0
      safeCall(cudaStreamCreate(&stream));
#else
      safeCall(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
#endif

      detectors.emplace_back(std::make_unique<DetectorContext>(
          siftParams, id, stream, collectCovariance));
      freeDetectors.push(detectors.back().get());
    }
  }
}
using namespace std::chrono_literals;

void CudaSift::detectAndExtract(const ImageU8 &image,
                                std::vector<SiftPoint> &siftPoints) {
  std::unique_lock<std::mutex> lock(mutex);
  while (!freeDetectors.size()) {
    detectorAvailable.wait(lock, [&]() { return freeDetectors.size() > 0; });
  }
  auto detector = freeDetectors.front();
  freeDetectors.pop();
  lock.unlock();

  detector->realloc(image.width, image.height, 0, image.is_bgr);

  safeCall(cudaSetDevice(detector->device));
  if (image.is_bgr) {
    memcpy(detector->u8_pinned, image.data, image.total());
    safeCall(cudaMemcpyAsync(detector->u8_device, detector->u8_pinned,
                             image.total(), cudaMemcpyHostToDevice,
                             (cudaStream_t)detector->stream));
    detector->processBgrU8(image.width, image.height, image.stride_bytes);
  } else {
    memcpy(detector->u8_pinned, image.data, image.total());
    safeCall(cudaMemcpyAsync(detector->u8_device, image.data, image.total(),
                             cudaMemcpyHostToDevice,
                             (cudaStream_t)detector->stream));
    detector->processU8(image.width, image.height, image.stride_bytes);
  }
  int nPts = detector->siftData->numPts;
  siftPoints.resize(nPts);
  memcpy(siftPoints.data(), detector->siftData->hostPtr(),
         sizeof(SiftPoint) * nPts);

  lock.lock();
  freeDetectors.push(detector);
  detectorAvailable.notify_all();
}

void CudaSift::detectAndExtract(const JPEGImage &image,
                                std::vector<SiftPoint> &siftPoints) {
  std::unique_lock<std::mutex> lock(mutex);
  if (!useNvJpeg || !freeDetectors.size()) {
    lock.unlock();
    auto tjState = tjInitDecompress();
    std::unique_ptr<uint8_t[]> data =
        std::make_unique<uint8_t[]>(image.width() * image.height());
    auto tjstatus = tjDecompress2(tjState, image.data(), image.size(),
                                  data.get(), image.width(), 0, image.height(),
                                  TJPF_GRAY, TJFLAG_FASTDCT);
    if (tjstatus && tjGetErrorCode(tjState) == TJERR_FATAL) {
      std::string str = "Failed to decompress " + image.name() +
                        " by turbo-jpeg : " + std::to_string(tjstatus) + " " +
                        tjGetErrorStr2(tjState);
      tjDestroy(tjState);

      throw std::runtime_error(str);
    } else {
      tjDestroy(tjState);
      detectAndExtract(ImageU8(data.get(), image.width(), image.height(), false,
                               image.width()),
                       siftPoints);
      return;
    }
  }
  while (!freeDetectors.size()) {
    detectorAvailable.wait(lock, [&]() { return freeDetectors.size() > 0; });
  }
  auto detector = freeDetectors.front();
  freeDetectors.pop();
  lock.unlock();

  detector->realloc(image.width(), image.height(), 0, false);

  safeCall(cudaSetDevice(detector->device));
  detector->processJPEG(image.data(), image.width(), image.height(),
                        image.size());
  int nPts = detector->siftData->numPts;
  siftPoints.resize(nPts);

  memcpy(siftPoints.data(), detector->siftData->hostPtr(),
         sizeof(SiftPoint) * nPts);

  lock.lock();
  freeDetectors.push(detector);
  detectorAvailable.notify_all();
}

void CudaSift::detectAndExtract(const ImageFP32 &image,
                                std::vector<SiftPoint> &siftPoints) {
  std::unique_lock<std::mutex> lock(mutex);
  while (!freeDetectors.size()) {
    detectorAvailable.wait(lock, [&]() { return freeDetectors.size() > 0; });
  }
  auto detector = freeDetectors.front();
  freeDetectors.pop();
  lock.unlock();

  detector->realloc(image.width, image.height, 0, false);
  safeCall(cudaSetDevice(detector->device));
  memcpy(detector->float_pinned, image.data, image.total());
  int targetStride =
      iAlignUp(image.width, SiftDetectorImpl::MIN_ALIGNMENT) * sizeof(float);

  safeCall(cudaMemcpy2DAsync(
      detector->float_device, targetStride, detector->float_pinned,
      image.stride_bytes, image.width * sizeof(float), image.height,
      cudaMemcpyHostToDevice, (cudaStream_t)detector->stream));
  detector->processFP32(image.width, image.height, targetStride);
  int nPts = detector->siftData->numPts;
  siftPoints.resize(nPts);
  memcpy(siftPoints.data(), detector->siftData->hostPtr(),
         sizeof(SiftPoint) * nPts);

  lock.lock();
  freeDetectors.push(detector);
  detectorAvailable.notify_all();
}

CudaSift::~CudaSift() {}

CudaSift::DetectorContext::~DetectorContext() {
  freeMemory();
  safeCall(cudaHostUnregister(detector.get()));
  safeCallNVJpeg(nvjpegJpegStateDestroy((nvjpegJpegState_t)nvjpegState));
  tjDestroy(tjState);
}
void CudaSift::DetectorContext::freeMemory() {
  alloc_u8 = alloc_float = alloc_features = 0;
  safeCall(cudaSetDevice(device));
  safeCall(cudaFree(float_device));
  safeCall(cudaFree(u8_device));
  safeCall(cudaFreeHost(float_pinned));
  safeCall(cudaFreeHost(u8_pinned));
}

CudaSift::DetectorContext::DetectorContext(const SiftParams &params,
                                           int deviceId, void *stream,
                                           bool collectCovariance,
                                           int width_init, int height_init)
    : detector(std::make_unique<SiftDetectorImpl>(params, deviceId,
                                                  (cudaStream_t)stream)),
      stream(stream), device(deviceId),
      covariance(collectCovariance ? new CovarianceEstimator(deviceId)
                                   : nullptr),
      collectCovariance(collectCovariance) {
  realloc(width_init, height_init, params.nFeatures, false);
  safeCall(cudaHostRegister(detector.get(), sizeof(SiftDetectorImpl),
                            cudaHostRegisterPortable));
  safeCallNVJpeg(nvjpegJpegStateCreate(NVJPEG::GetHandle(),
                                       (nvjpegJpegState_t *)&nvjpegState));
  tjState = tjInitDecompress();
}

void CudaSift::getResults(double *mean, double *covariance) {
  for (int i = 1; i < detectors.size(); ++i)
    detectors[0]->covariance->mergeWith(detectors[i]->covariance.get());
  detectors[0]->covariance->getResults(mean, covariance);
}

void CudaSift::DetectorContext::realloc(int w, int h, int nFeatures,
                                        bool need_rgb) {
  int padded = iAlignUp(w, SiftDetectorImpl::MIN_ALIGNMENT);
  int need_size = (need_rgb ? 3 : 1) * padded * h;
  int need_float = padded * h;
  if (need_size > alloc_u8 || need_float > alloc_float) {
    freeMemory();
    alloc_u8 = need_size;
    alloc_float = need_float;

    safeCall(cudaSetDevice(device));
    safeCall(
        cudaHostAlloc((void **)&u8_pinned, need_size, cudaHostAllocPortable));
    safeCall(cudaHostAlloc((void **)&float_pinned, sizeof(float) * need_float,
                           cudaHostAllocPortable));
    safeCall(cudaMalloc((void **)&float_device, sizeof(float) * need_float));
    safeCall(cudaMalloc((void **)&u8_device, need_size));
  }

  if (nFeatures > alloc_features) {
    siftData = std::make_unique<SiftData>();
    siftData->allocate(nFeatures, true, true, device);
    alloc_features = nFeatures;
  }
}

} // namespace cudasift
