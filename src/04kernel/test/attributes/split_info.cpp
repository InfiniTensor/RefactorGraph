#include "kernel/attributes/split_info.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

TEST(kernel, SplitInfo) {
    std::vector<Arc<Tensor>> outputs{
        Tensor::share(DataType::F32, Shape{2, 3, 1, 1, 7, 7}),
        Tensor::share(DataType::F32, Shape{2, 3, 1, 9, 7, 7}),
        Tensor::share(DataType::F32, Shape{2, 3, 1, 3, 7, 7}),
        Tensor::share(DataType::F32, Shape{2, 3, 1, 7, 7, 7}),
    };
    TensorRefs outputs_;
    outputs_.reserve(outputs.size());
    std::transform(outputs.begin(), outputs.end(),
                   std::back_inserter(outputs_),
                   [](auto const &it) { return std::cref(*it); });
    SplitInfo info(3, outputs_);
    std::equal(info.prefix().begin(), info.prefix().end(),
               std::array<uint_lv2, 3>{2, 3, 1}.begin());
    std::equal(info.postfix().begin(), info.postfix().end(),
               std::array<uint_lv2, 4>{1 * 49, 9 * 49, 3 * 49, 7 * 49}.begin());
}
