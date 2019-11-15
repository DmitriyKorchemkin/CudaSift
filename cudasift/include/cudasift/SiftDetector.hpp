#ifndef SIFTDETECTOR_HPP
#define SIFTDETECTOR_HPP

#include <vector>
#include <iostream>

#include "cudaSift.h"

namespace cudasift {


struct DescriptorNormalizer {
    struct NormalizerStep {
        NormalizerStep(const NormalizerOp &op, std::vector<float> data = {});
        template <typename T>
        NormalizerStep(const NormalizerOp &op, const T& t) : NormalizerStep(op, std::vector<float>(t)) {}

        NormalizerOp op() const;
        const std::vector<float>& data() const;
        private:
        NormalizerOp _op;
        std::vector<float> _data;
    };

    DescriptorNormalizer(std::istream& stream);
    DescriptorNormalizer(std::vector<NormalizerStep> steps);
    DescriptorNormalizer();

    DescriptorNormalizerData exportNormalizer() const;

    std::vector<NormalizerStep> steps;
    mutable std::vector<float> data;
    mutable std::vector<int> steps_i;
};

struct SiftParams {
    int nFeatures;

    int numOctaves;
    float initBlur;
    float threshold;

    float lowestScale;
    bool scaleUp = false;

    DescriptorNormalizer normalizer;
};

struct CudaSift {
    CudaSift(const SiftParams siftParams = SiftParams(), int deviceId = 0, void* stream = nullptr);
    ~CudaSift();
    private:
        int deviceId;
        void* stream;
};

}

#endif
