//********************************************************//
// CUDA SIFT extractor by Marten Björkman aka Celebrandil //
//              celle @ csc.kth.se                       //
//********************************************************//

#include <cmath>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
  // cv::flip(limg, rimg, -1);
  unsigned int w = limg.cols;
  unsigned int h = limg.rows;
  std::cout << "Image size = (" << w << "," << h << ")" << std::endl;

  // Initial Cuda images and download images to device
  std::cout << "Initializing data..." << std::endl;
  InitCuda(devNum);
  CudaImage img1, img2;
  img1.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float *)limg.data);
  img2.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float *)rimg.data);
  img1.Download();
  img2.Download();

  // Extract Sift features from images
  SiftData siftData1, siftData2;
  float initBlur = 1.0f;
  float thresh = (imgSet ? 4.5f : 3.0f);
  InitSiftData(siftData1, 32768, true, true);
  InitSiftData(siftData2, 32768, true, true);

  DescriptorNormalizerData data;
  data.n_steps = 5;
  data.n_data = 1;
  int steps[] = {1, 4, 1, 3, 0};
  float dataf[] = {0.2f};
  data.normalizer_steps = steps;
  data.data = dataf;

  const int N = 10000;
  std::vector<double> detect, match;

  // A bit of benchmarking
  // for (int thresh1=1.00f;thresh1<=4.01f;thresh1+=0.50f) {
  float *memoryTmp = AllocSiftTempMemory(w, h, 5, false);
  for (int i = 0; i < N; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    ExtractSift(siftData1, img1, 5, initBlur, thresh, data, 0.0f, false,
                memoryTmp);
    auto stop1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff1 = stop1 - start;
    double delta;
    delta = diff1.count();
    detect.push_back(delta);
    ExtractSift(siftData2, img2, 5, initBlur, thresh, data, 0.0f, false,
                memoryTmp);
    auto stop2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff2 = stop2 - stop1;
    delta = diff2.count();
    detect.push_back(delta);
  }
  FreeSiftTempMemory(memoryTmp);

  // Match Sift features and find a homography
  for (int i = 0; i < N; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    MatchSiftData(siftData1, siftData2);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = stop - start;
    double delta = diff.count();
    match.push_back(delta);
  }
  float homography[9];
  int numMatches;
  std::cout << "Number of original features: " << siftData1.numPts << " "
            << siftData2.numPts << std::endl;

  std::sort(detect.begin(), detect.end());
  std::sort(match.begin(), match.end());
  double med;
  med = detect[0.5 * N];
  std::cout << "Detect: [" << med << " -" << (med - detect[0.0005 * N]) << " +"
            << (detect[0.9995 * N] - med) << "]" << std::endl;
  med = match[0.5 * N];
  std::cout << "Match: [" << med << " -" << (med - match[0.0005 * N]) << " +"
            << (match[0.9995 * N] - med) << "]" << std::endl;

  FreeSiftData(siftData1);
  FreeSiftData(siftData2);
}
