#include "cudasift/SiftDetector.hpp"

#include <limits>
#include <type_traits>

namespace cudasift {

    DescriptorNormalizer::NormalizerStep::NormalizerStep(const NormalizerOp &op, std::vector<float> data) : _op(op), _data(std::move(data)) {
    if (OpDataSize(_op) != _data.size())
        throw std::runtime_error("Invalid payload size (" + std::to_string(_data.size()) +  ", expected: " + std::to_string(OpDataSize(_op)) + ")");
}

NormalizerOp DescriptorNormalizer::NormalizerStep::op() const {
    return _op;
}

const std::vector<float> & DescriptorNormalizer::NormalizerStep::data() const {
    return _data;
}

DescriptorNormalizer::DescriptorNormalizer(std::istream& stream) {
    uint32_t n_steps;
    stream >> n_steps;

    for (int i = 0; i < n_steps; ++i) {
        std::underlying_type<NormalizerOp>::type op_id;
        stream >> op_id;
        auto op = static_cast<NormalizerOp>(op_id);
        auto size = OpDataSize(op);
        if (size == std::numeric_limits<size_t>::max())
            throw std::runtime_error("Invalid normalizer operation: " + std::to_string(op_id));

        std::vector<float> vec(size);
        for (int j = 0; j < size; ++j)
            stream >> vec[j];

        steps.emplace_back(op, std::move(vec));
            
            

    }
}


DescriptorNormalizer::DescriptorNormalizer(std::vector<NormalizerStep> steps) : steps(std::move(steps)) {
}

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

    for (auto &step: steps) {
        const auto& payload = step.data();
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


}

