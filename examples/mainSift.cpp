//********************************************************//
// CUDA SIFT extractor by Marten Björkman aka Celebrandil //
//              celle @ csc.kth.se                       //
//********************************************************//  

#include <iostream>  
#include <cmath>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cudasift/cudaImage.h"
#include "cudasift/cudaSift.h"

void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img);
void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography);

double ScaleUp(CudaImage &res, CudaImage &src);


///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) 
{    
  int devNum = 0, imgSet = 0;
  if (argc>1)
    devNum = std::atoi(argv[1]);
  if (argc>2)
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
  //cv::flip(limg, rimg, -1);
  unsigned int w = limg.cols;
  unsigned int h = limg.rows;
  std::cout << "Image size = (" << w << "," << h << ")" << std::endl;
  
  // Initial Cuda images and download images to device
  std::cout << "Initializing data..." << std::endl;
  InitCuda(devNum); 
  CudaImage img1, img2;
  img1.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)limg.data);
  img2.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)rimg.data);
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

  // A bit of benchmarking 
  //for (int thresh1=1.00f;thresh1<=4.01f;thresh1+=0.50f) {
  float *memoryTmp = AllocSiftTempMemory(w, h, 5, false);
    for (int i=0;i<1000;i++) {
      ExtractSift(siftData1, img1, 5, initBlur, thresh, data, 0.0f, false,
                  memoryTmp);
      ExtractSift(siftData2, img2, 5, initBlur, thresh, data, 0.0f, false,
                  memoryTmp);
    }
    FreeSiftTempMemory(memoryTmp);
    
    // Match Sift features and find a homography
    for (int i=0;i<1;i++)
      MatchSiftData(siftData1, siftData2);
    float homography[9];
    int numMatches;
    std::cout << "Number of original features: " <<  siftData1.numPts << " " << siftData2.numPts << std::endl;
    
    cv::imwrite("data/limg_pts.pgm", limg);

  // Free Sift data from device
  FreeSiftData(siftData1);
  FreeSiftData(siftData2);
}
