#include "cudasift/SiftDetector.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#include <cxxopts.hpp>

namespace fs = std::filesystem;

void memcpy_pitch(void *dst, size_t dst_pitch, void *src, size_t src_pitch,
                  size_t height, size_t width) {
  if (src_pitch == dst_pitch && width == src_pitch) {
    memcpy(dst, src, src_pitch * height);
    return;
  }
  for (int i = 0; i < height; ++i)
    memcpy((uint8_t *)dst + dst_pitch * i, (uint8_t *)src + src_pitch * i,
           width);
}

size_t save(const std::string &output_name, bool saveOnlyDescriptors,
            const std::vector<cudasift::SiftPoint> &res) {
  int fd = open(output_name.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0666);
  size_t size_single =
      saveOnlyDescriptors ? 128 * sizeof(float) : sizeof(cudasift::SiftPoint);

  auto size = res.size() * size_single;
  if (ftruncate(fd, size)) {
    perror("Failed to call ftruncate");
  }
  void *mapped = mmap(nullptr, size, PROT_WRITE, MAP_SHARED, fd, 0);
  if (mapped == (void *)~((uintptr_t)0)) {
    perror("Failed to call mmap");
  }
  memcpy_pitch(mapped, size_single,
               saveOnlyDescriptors ? (void *)&res[0].data[0] : (void *)&res[0],
               sizeof(cudasift::SiftPoint), res.size(), size_single);
  munmap(mapped, size);
  close(fd);
  return size;
}

int main(int argc, char **argv) {
  std::vector<std::pair<std::string, std::string>> files;
  cxxopts::Options options(argv[0], " - bulk detection of *SIFT features");
  // clang-format off
  options.add_options()
      ("h,help", "Help")
      ("config", "Descriptor normalizer config", cxxopts::value<std::string>())
      ("file-list","List of input [output] files (two file names per line if output is required, one file name per line otherwise)", cxxopts::value<std::string>())
      ("save-data", "Save detection results", cxxopts::value<bool>())
      ("save-only-descriptors", "Omit feature point data when saving", cxxopts::value<bool>())
      ("updated-config", "Output config (with mean substraction & whitening transformation; keep in mind that you will need to add normalization in order to use with correlation-based matcher)", cxxopts::value<std::string>())
      ("collect-stats", "Collect stats (mean & covariance)", cxxopts::value<std::string>())
      ("export-stats", "Filename for stats", cxxopts::value<std::string>())
      ("normalization-type", "Normalization transform type (after zca/pca: none, l1, l2; l1 transformation includes sqrt [i.e. to correlation-compatible form])", cxxopts::value<std::string>())
      ("whitening-type", "Whitening transform type (zca or pca)", cxxopts::value<std::string>())
      ("clip-explained-variance", "Explained variance threshold", cxxopts::value<double>())
      ("n-threads", "CPU threads to schedule", cxxopts::value<int>())
      ("devices", "GPUs to run on", cxxopts::value<std::vector<int>>())
      ("streams-per-device", "Streams per device", cxxopts::value<int>())
      ("use-nvJPEG", "Use nvJPEG for image decoding", cxxopts::value<bool>());

  // clang-format on
  auto result = options.parse(argc, argv);
  std::string input_list;
  if (result.count("h")) {
    std::cout << options.help() << '\n';
    return 0;
  }
  if (!result.count("file-list")) {
    std::cerr << "File list is required\n";
    exit(-1);
  } else {
    input_list = result["file-list"].as<std::string>();
    std::ifstream ifs(input_list);
    if (!ifs) {
      std::cerr << "Invalid file list file " << input_list << '\n';
      exit(-2);
    }
    std::string str, name_in, name_out;
    while (ifs) {
      std::getline(ifs, str);
      std::stringstream ss(str);
      ss >> name_in >> name_out;
      if (ifs)
        files.emplace_back(name_in, name_out);
    }
  }
  cudasift::SiftParams params;
  if (result.count("config")) {
    std::string config_name = result["config"].as<std::string>();
    std::ifstream params_stream(config_name);
    params.normalizer = cudasift::DescriptorNormalizer(params_stream);
  }

  bool collectStats = false, saveData = false, saveOnlyDescriptors = false;
  if (result.count("save-data")) {
    saveData = result["save-data"].as<bool>();
  }
  if (result.count("save-only-descriptors")) {
    saveOnlyDescriptors = result["save-only-descriptors"].as<bool>();
  }
  if (result.count("collect-stats")) {
    collectStats = result["collect-stats"].as<bool>();
  }
  std::string output_config_name;
  if (result.count("updated-config")) {
    collectStats = true;
    output_config_name = result["updated-config"].as<std::string>();
  }
  std::string output_stats_name;
  if (result.count("export-stats")) {
    collectStats = true;
    output_stats_name = result["export-stats"].as<std::string>();
  }
  bool zca = true;
  if (result.count("whitening-type")) {
    std::string str = result["whitening-type"].as<std::string>();
    if (str == "zca") {
      zca = true;
    } else if (str == "pca") {
      zca = false;
    } else {
      std::cout << "Unsupported whitening type: " << str << '\n';
      exit(-1);
    }
  }
  int append_normalization = -1;
  if (result.count("normalization-type")) {
    std::string str = result["normalization-type"].as<std::string>();
    if (str == "l1") {
      append_normalization = 0;
    } else if (str == "l2") {
      append_normalization = 1;
    } else if (str == "none") {
      append_normalization = -1;
    } else {
      std::cout << "Unsupported normalization type: " << str << '\n';
      exit(-1);
    }
  }
  double unexplained_varaiance_threshold = 0.005;
  if (result.count("clip-unexplained-variance")) {
    unexplained_varaiance_threshold =
        result["clip-unexplained-variance"].as<double>();
    unexplained_varaiance_threshold =
        std::max(0., std::min(1., unexplained_varaiance_threshold));
  }
  bool useNvJpeg = false;
  if (result.count("use-nvJPEG")) {
    useNvJpeg = result["use-nvJPEG"].as<bool>();
  }
  int Nthreads = std::thread::hardware_concurrency();
  if (result.count("n-threads")) {
    Nthreads = result["n-threads"].as<int>();
  }

  int streamsPerDevice = 6;
  std::vector<int> devices;
  if (result.count("streams-per-device")) {
    streamsPerDevice = result["streams-per-device"].as<int>();
  }
  if (result.count("devices")) {
    devices = result["devices"].as<std::vector<int>>();
  }

  cudasift::CudaSift multiDetector(params, devices, streamsPerDevice,
                                   collectStats, useNvJpeg);
  int N = files.size();

  std::atomic<int> ai(0);
  std::atomic<size_t> bytes(0);
  auto start = std::chrono::high_resolution_clock::now();

  std::vector<std::thread> threads;
  for (int i = 0; i < Nthreads; ++i) {
    threads.emplace_back([&]() {
      std::vector<cudasift::SiftPoint> res;
      while (ai < N) {
        int id = ai++;
        if (ai >= N)
          break;
        try {
          cudasift::JPEGImage image(files[id].first.c_str());
          try {
            multiDetector.detectAndExtract(image, res);
            if (res.size()) {
              if (saveData) {
                bytes += save(files[id].second, saveOnlyDescriptors, res);
              }
            } else {
              std::cout << "0 features for " << files[id].first << std::endl;
            }

          } catch (const std::runtime_error &err) {
            std::cout << "Failure: " << err.what() << " " << files[id].first
                      << std::endl;
          }
        } catch (const std::runtime_error &err) {
          std::cout << "Failed to decode image header (" + files[id].first + ")"
                    << std::endl;
        }
        int a = ++ai;
        if (a % ((N + 99) / 100) == 0) {
          auto stop = std::chrono::high_resolution_clock::now();
          double gbytes = bytes / 1024. / 1024 / 1024;
          double dt = 1e9 / (stop - start).count();
          std::cout << double(a) / N * 100. << "% (" << gbytes
                    << "GBytes written [" << gbytes * dt << "Gb/s] " << (a * dt)
                    << " FPS)" << std::endl;
        }
      }
    });
  }
  for (auto &t : threads)
    t.join();

  auto stop = std::chrono::high_resolution_clock::now();
  std::cout << (N * 1e9 / (stop - start).count()) << " FPS" << std::endl;

  cv::Mat cov(128, 128, CV_64FC1);
  cv::Mat mean(1, 128, CV_64FC1);
  cov = -1.;
  mean = -1.;
  if (collectStats) {
    multiDetector.getResults(mean.ptr<double>(), cov.ptr<double>());

    if (output_stats_name.size()) {
      std::ofstream of(output_stats_name);
      of << "M = " << mean << ';' << std::endl
         << "S = " << cov << ';' << std::endl;
    }

    cv::Mat values, vectorst;
    cv::eigen(cov, values, vectorst);
    cv::Mat invDiag = cv::Mat::zeros(128, 128, CV_64FC1);
    cv::Mat diag = cv::Mat::zeros(128, 128, CV_64FC1);
    double total = 0.;
    for (int i = 0; i < 128; ++i) {
      double ev = values.at<double>(i);
      total += ev;
      diag.at<double>(i, i) = ev;
    }
    total *= (1. - unexplained_varaiance_threshold);
    for (int i = 0; i < 128; ++i) {
      double ev = values.at<double>(i);
      invDiag.at<double>(i, i) = 1. / std::sqrt(ev);
      total -= ev;
      if (total < 0.) {
        std::cout << "Discarded eigenvalues starting from " << i + 1
                  << std::endl;
        break;
      }
    }
    double maxOffdiag = 0.;
    for (int i = 0; i < 128; ++i) {
      for (int j = 0; j < 128; ++j) {
        if (i == j)
          continue;
        double cov_v = std::abs(cov.at<double>(i, j));
        maxOffdiag = std::max(maxOffdiag, cov_v);
      }
    }
    std::cout << "Eigen-values: " << values.t() / values.at<double>(0) << '\n';
    std::cout << "Min-max ratio: "
              << values.at<double>(0) / values.at<double>(127) << '\n';
    std::cout << "Max covariance: " << maxOffdiag << '\n';

    cv::Mat normalizer;
    if (zca) {
      normalizer = vectorst.t() * invDiag * vectorst;
    } else {
      normalizer = invDiag * vectorst;
    }
    std::vector<float> sub_mean(128), rot(128 * 128);
    int idx = 0;
    for (int i = 0; i < 128; ++i) {
      sub_mean[i] = -mean.at<double>(i);
      for (int j = 0; j < 128; ++j) {
        rot[idx++] = normalizer.at<double>(i, j);
      }
    }
    params.normalizer.steps.pop_back();
    params.normalizer.steps.emplace_back(cudasift::Add, sub_mean);
    params.normalizer.steps.emplace_back(cudasift::Mul, rot);

    switch (append_normalization) {
    case 0:
      params.normalizer.steps.emplace_back(cudasift::ComputeL1);
      params.normalizer.steps.emplace_back(cudasift::DivideByNorm);
      params.normalizer.steps.emplace_back(cudasift::Sqrt);
      break;
    case 1:
      params.normalizer.steps.emplace_back(cudasift::ComputeL2);
      params.normalizer.steps.emplace_back(cudasift::DivideByNorm);
      break;
    default:
      break;
    }

    params.normalizer.steps.emplace_back(cudasift::CopyToOutput);
  }

  if (output_config_name.size()) {
    std::ofstream nout(output_config_name);
    params.normalizer.exportNormalizer(nout);
  }
  return 0;
}
