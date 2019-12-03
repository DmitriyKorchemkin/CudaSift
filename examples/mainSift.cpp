//********************************************************//
// CUDA SIFT extractor by Marten Björkman aka Celebrandil //
//              celle @ csc.kth.se                       //
//********************************************************//

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <thread>
#include <mutex>

#include "cudasift/cudaImage.h"
#include "cudasift/cudaSift.h"

using namespace cudasift;

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  int devNum = 0, imgSet = 0;
  if (argc > 1)
    devNum = std::atoi(argv[1]);
  if (argc > 2)
    imgSet = std::atoi(argv[2]);

  // Read images using OpenCV
  cv::Mat limg, rimg;
  if (imgSet) {
    cv::imread("data/left.pgm", 0).convertTo(limg, CV_32FC1);
    cv::imread("data/righ.pgm", 0).convertTo(rimg, CV_32FC1);
  } else {
    cv::imread("data/img1.png", 0).convertTo(limg, CV_32FC1);
    cv::imread("data/img2.png", 0).convertTo(rimg, CV_32FC1);
  }
  unsigned int w = limg.cols;
  unsigned int h = limg.rows;
  std::cout << "Image size = (" << w << "," << h << ")" << std::endl;

  // Initial Cuda images and download images to device
  std::cout << "Initializing data..." << std::endl;
  InitCuda(devNum);

  int nDevice;
  cudaGetDeviceCount(&nDevice);
  int nStream = 2;
  int nDetectors = nDevice * nStream;
  // Extract Sift features from images
  std::vector<CudaImage> imgs(nDetectors);
  std::vector<SiftData> siftData(nDetectors);
  std::vector<cudaStream_t> streams(nDetectors);
  std::vector<std::unique_ptr<SiftDetectorImpl>> detectors(nDetectors);
  std::vector<std::thread> threads;
  int idx = 0;
  for (int d = 0; d < nDevice; ++d) {
    cudaSetDevice(d);
    for (int i = 0; i < nStream; ++i) {
      siftData[idx].allocate(32768, true, true, d);
      cudaStreamCreate(&streams[idx]);
      detectors[idx] =
          std::make_unique<SiftDetectorImpl>(SiftParams(), d, streams[idx]);
      imgs[idx].Allocate(w, h, iAlignUp(w, 128), false, NULL,
                         (float *)(idx % 2 ? rimg : limg).data);
      imgs[idx].Download();
      ++idx;
    }
  }

  const int N = 100000;
  std::vector<double> detect, match;
  std::mutex mdetect;

  auto startD = std::chrono::high_resolution_clock::now();
  for (int ii = 0; ii < nDetectors; ++ii) {
    threads.emplace_back([&, ii]() {
      int id = ii;
      for (int i = 0; i < N; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        detectors[id]->ExtractSift(siftData[id], imgs[id]);
        auto stop1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff1 = stop1 - start;
        double delta;
        delta = diff1.count();
        std::unique_lock<std::mutex> lock(mdetect);
        detect.push_back(delta);
      }
    });
  }

  for (auto &t : threads)
    t.join();
  auto stopD = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diffD = stopD - startD;
  std::cout << "Detected SIFT features on " << nDetectors * N << " images in "
            << diffD.count() << " second (" << (nDetectors * N) / diffD.count()
            << "FPS)" << std::endl;

  // Match Sift features and find a homography
  cudaSetDevice(0);
  for (int i = 0; i < N; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    MatchSiftData(siftData[0], siftData[1]);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = stop - start;
    double delta = diff.count();
    match.push_back(delta);
  }
  float homography[9];
  int numMatches;
  std::cout << "Number of original features: " << siftData[0].numPts << " "
            << siftData[1].numPts << std::endl;

  std::sort(detect.begin(), detect.end());
  std::sort(match.begin(), match.end());
  double med;
  med = detect[0.5 * N];
  std::cout << "Detect: [" << med << " -" << (med - detect[0.0005 * N]) << " +"
            << (detect[0.9995 * N] - med) << "]" << std::endl;
  med = match[0.5 * N];
  std::cout << "Match: [" << med << " -" << (med - match[0.0005 * N]) << " +"
            << (match[0.9995 * N] - med) << "]" << std::endl;

  return 0;
}
