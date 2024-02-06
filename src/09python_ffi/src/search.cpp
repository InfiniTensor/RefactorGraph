#include "search.h"
#include "functions.h"
#include <random>

namespace refactor::python_ffi {

    pybind11::array randomSearch(pybind11::array logits_, int topK, float topP, float temperature) {
        auto shape = std::span(logits_.shape(), logits_.ndim());
        ASSERT(!shape.empty(), "");
        auto shapeBack = shape.begin() + shape.size() - 1;
        auto batch = std::accumulate(shape.begin(), shapeBack, 1l, std::multiplies()),
             vocabSize = *shapeBack;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> rng(0, 1);
        std::vector<int64_t> result(batch);
        for (auto i : range0_(batch)) {
            // cast
            std::vector<float> logits(vocabSize);
            if (auto type = parseNumpyDType(logits_.dtype()); type == DataType::FP16) {
                auto data = reinterpret_cast<fp16_t const *>(logits_.data()) + i * vocabSize;
                std::transform(data, data + logits.size(), logits.begin(), [=](auto x) { return x.to_f32() / temperature; });
            } else if (type == DataType::F32) {
                auto data = reinterpret_cast<float const *>(logits_.data()) + i * vocabSize;
                std::transform(data, data + logits.size(), logits.begin(), [=](auto x) { return x / temperature; });
            } else {
                RUNTIME_ERROR("unsupported data type.");
            }
            std::vector<std::pair<float, int>> probabilities(vocabSize);
            // softmax
            for (auto max = *std::max_element(logits.begin(), logits.end());
                 auto j : range0_(vocabSize)) {
                auto p = std::exp(logits[j] - max);
                probabilities[j] = {p, j};
            }
            auto k = 0;
            {// topK + topP
                std::sort(probabilities.begin(), probabilities.end(),
                          [](auto a, auto b) { return a.first > b.first; });
                for (auto cum = 0.f; auto i : range0_(topK)) {
                    ++k;
                    if ((cum += probabilities[i].first) > topP) {
                        break;
                    }
                }
            }
            auto chosen = false;
            auto p = rng(gen);
            // re-softmax
            for (auto max = probabilities[0].first;
                 auto j : range0_(k)) {
                auto p_ = std::exp(probabilities[j].first - max);
                if (p_ >= p) {
                    result[i] = probabilities[j].second;
                    chosen = true;
                    break;
                }
                p -= p_;
            }
            if (!chosen) {
                result[i] = probabilities[k - 1].second;
            }
        }

        return pybind11::array(buildNumpyDType(DataType::I64), std::span(shape.begin(), shape.size() - 1), result.data());
    }

}// namespace refactor::python_ffi
