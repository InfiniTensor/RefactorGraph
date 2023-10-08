#include "frontend/infer.h"
#include "common/natural.h"
#include <execution>

namespace refactor::frontend {

    InferError::InferError(std::string msg)
        : std::runtime_error(std::move(msg)),
          value(FatalError{}) {}
    InferError::InferError(UnknownVariable variable)
        : std::runtime_error(fmt::format("Unknown variable: {}", variable.name)),
          value(std::move(variable)) {}

    bool InferOptions::shouldCalculate(
        TensorRefs inputs,
        std::vector<std::reference_wrapper<Tensor const>> outputs) const {
        if (!calculate) { return false; }
        size_t sizeI = 0;
        for (auto const &input : inputs) {
            if (!input.hasData()) {
                return false;
            }
            if (!input.depVariables.empty()) {
                sizeI += input.bytesSize();
            }
        }
        auto sizeO = std::accumulate(outputs.begin(), outputs.end(), 0ul,
                                     [](auto const &acc, auto const &output) {
                                         return acc + output.get().depVariables.empty()
                                                    ? 0
                                                    : output.get().bytesSize();
                                     });
        return sizeO < std::max(bytesDilationThreshold * sizeI, calculationByteThreshold);
    }

    std::unordered_set<DimVariable> extractDependency(TensorRefs inputs) {
        std::unordered_set<DimVariable> ans;
        std::for_each_n(common::natural_t(0), inputs.size(),
                        [&inputs, &ans](auto const i) {
                            Tensor const &input = inputs[i];
                            for (auto const &dim : input.shape) {
                                if (dim.isVariable()) {
                                    ans.insert(dim.variable());
                                }
                            }
                            for (auto var : input.depVariables) {
                                ans.insert(var);
                            }
                        });
        return ans;
    }

    SmallInts<4> locateN(Shape const &shape, size_t k) {
        SmallInts<4> indices(shape.size());
        auto i = indices.rbegin();
        auto j = shape.rbegin(),
             ej = shape.rend();
        while (j != ej) {
            auto div = std::div(k, j++->value());
            *i++ = div.rem;
            k = div.quot;
        }
        return indices;
    }

    void const *locate1(Tensor const &tensor, SmallInts<4> const &indices) {
        auto i = indices.rbegin();
        auto j = tensor.shape.rbegin(),
             ej = tensor.shape.rend();
        size_t k = 0, mul = 1;
        // while (i != ei && j != ej) { // 广播不可能使形状缩短
        while (j != ej) {
            auto const shape = j++->value();
            auto const i_ = *i++;
            k += shape == 1 ? 0 : i_ * mul;
            mul *= shape;
        }
        return tensor.data->get<uint8_t>() + k * tensor.dataType.size();
    }
}// namespace refactor::frontend
