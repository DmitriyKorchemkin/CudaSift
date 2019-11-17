#include "cudasift/cudaSift.h"
#include "cudasift/cudautils.h"

//================= Device matching functions =====================//

namespace cudasift {

__global__ void CleanMatches(SiftPoint *sift1, int numPts1) {
  const int p1 = min(blockIdx.x * 64 + threadIdx.x, numPts1 - 1);
  sift1[p1].score = 0.0f;
}

#define M7W 32
#define M7H 32
#define M7R 4
#define NRX 2
#define NDIM 128

__global__ void FindMaxCorr10(SiftPoint *sift1, SiftPoint *sift2, int numPts1,
                              int numPts2) {
  __shared__ float4 buffer1[M7W * NDIM / 4];
  __shared__ float4 buffer2[M7H * NDIM / 4];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bp1 = M7W * blockIdx.x;
  for (int j = ty; j < M7W; j += M7H / M7R) {
    int p1 = min(bp1 + j, numPts1 - 1);
    for (int d = tx; d < NDIM / 4; d += M7W)
      buffer1[j * NDIM / 4 + (d + j) % (NDIM / 4)] =
          ((float4 *)&sift1[p1].data)[d];
  }

  float max_score[NRX];
  float sec_score[NRX];
  int index[NRX];
  for (int i = 0; i < NRX; i++) {
    max_score[i] = 0.0f;
    sec_score[i] = 0.0f;
    index[i] = -1;
  }
  int idx = ty * M7W + tx;
  int ix = idx % (M7W / NRX);
  int iy = idx / (M7W / NRX);
  for (int bp2 = 0; bp2 < numPts2 - M7H + 1; bp2 += M7H) {
    for (int j = ty; j < M7H; j += M7H / M7R) {
      int p2 = min(bp2 + j, numPts2 - 1);
      for (int d = tx; d < NDIM / 4; d += M7W)
        buffer2[j * NDIM / 4 + d] = ((float4 *)&sift2[p2].data)[d];
    }
    __syncthreads();

    if (idx < M7W * M7H / M7R / NRX) {
      float score[M7R][NRX];
      for (int dy = 0; dy < M7R; dy++)
        for (int i = 0; i < NRX; i++)
          score[dy][i] = 0.0f;
      for (int d = 0; d < NDIM / 4; d++) {
        float4 v1[NRX];
        for (int i = 0; i < NRX; i++)
          v1[i] = buffer1[((M7W / NRX) * i + ix) * NDIM / 4 +
                          (d + (M7W / NRX) * i + ix) % (NDIM / 4)];
        for (int dy = 0; dy < M7R; dy++) {
          float4 v2 = buffer2[(M7R * iy + dy) * (NDIM / 4) + d];
          for (int i = 0; i < NRX; i++) {
            score[dy][i] += v1[i].x * v2.x;
            score[dy][i] += v1[i].y * v2.y;
            score[dy][i] += v1[i].z * v2.z;
            score[dy][i] += v1[i].w * v2.w;
          }
        }
      }
      for (int dy = 0; dy < M7R; dy++) {
        for (int i = 0; i < NRX; i++) {
          if (score[dy][i] > max_score[i]) {
            sec_score[i] = max_score[i];
            max_score[i] = score[dy][i];
            index[i] = min(bp2 + M7R * iy + dy, numPts2 - 1);
          } else if (score[dy][i] > sec_score[i])
            sec_score[i] = score[dy][i];
        }
      }
    }
    __syncthreads();
  }

  float *scores1 = (float *)buffer1;
  float *scores2 = &scores1[M7W * M7H / M7R];
  int *indices = (int *)&scores2[M7W * M7H / M7R];
  if (idx < M7W * M7H / M7R / NRX) {
    for (int i = 0; i < NRX; i++) {
      scores1[iy * M7W + (M7W / NRX) * i + ix] = max_score[i];
      scores2[iy * M7W + (M7W / NRX) * i + ix] = sec_score[i];
      indices[iy * M7W + (M7W / NRX) * i + ix] = index[i];
    }
  }
  __syncthreads();

  if (ty == 0) {
    float max_score = scores1[tx];
    float sec_score = scores2[tx];
    int index = indices[tx];
    for (int y = 0; y < M7H / M7R; y++)
      if (index != indices[y * M7W + tx]) {
        if (scores1[y * M7W + tx] > max_score) {
          sec_score = max(max_score, sec_score);
          max_score = scores1[y * M7W + tx];
          index = indices[y * M7W + tx];
        } else if (scores1[y * M7W + tx] > sec_score)
          sec_score = scores1[y * M7W + tx];
      }
    sift1[bp1 + tx].score = max_score;
    sift1[bp1 + tx].match = index;
    sift1[bp1 + tx].match_xpos = sift2[index].xpos;
    sift1[bp1 + tx].match_ypos = sift2[index].ypos;
    sift1[bp1 + tx].ambiguity = sec_score / (max_score + 1e-6f);
  }
}

double MatchSiftData(SiftData &data1, SiftData &data2) {
  TimerGPU timer(0);
  int numPts1 = data1.numPts;
  int numPts2 = data2.numPts;
  if (!numPts1 || !numPts2)
    return 0.0;
#ifdef MANAGEDMEM
  SiftPoint *sift1 = data1.m_data;
  SiftPoint *sift2 = data2.m_data;
#else
  if (data1.d_data == NULL || data2.d_data == NULL)
    return 0.0f;
  SiftPoint *sift1 = data1.d_data;
  SiftPoint *sift2 = data2.d_data;
#endif

  // Combined version with no global memory requirement using global locks
  dim3 blocksMax3(iDivUp(numPts1, 16), iDivUp(numPts2, 512));
  dim3 threadsMax3(16, 16);
  CleanMatches<<<iDivUp(numPts1, 64), 64>>>(sift1, numPts1);
  blocksMax3 = dim3(iDivUp(numPts1, M7W));
  threadsMax3 = dim3(M7W, M7H / M7R);
  FindMaxCorr10<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  safeCall(cudaDeviceSynchronize());

  double gpuTime = timer.read();
#ifndef VERBOSE
  printf("MatchSiftData time =          %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

} // namespace cudasift
