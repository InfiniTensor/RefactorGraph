#include "kernel/attributes/mat_mul_integer_info.h"

namespace refactor::kernel {

    MatMulIntegerInfo::Input::Input(TensorRefs const &inputs, size_t i) noexcept
        : withZeroPoint(false),
          signed_(true),
          scalar(true) {
        if (inputs.size() > i + 2) {
            auto const &t = inputs[i + 2].get();
            signed_ = t.dataType == DataType::I8;

            auto size = t.elementsSize();
            scalar = size == 1;

            if (t.data) {
                auto data = slice(t.data->get<uint8_t>(), size);
                withZeroPoint = std::any_of(data.begin(), data.end(), [](auto x) { return x != 0; });
            } else {
                withZeroPoint = true;
            }
        }
    }

    MatMulIntegerInfo::MatMulIntegerInfo(TensorRefs const &inputs) noexcept
        : a(inputs, 0),
          b(inputs, 1),
#define A (inputs[0].get().shape)
#define B (inputs[1].get().shape)
          m(A.rbegin()[1]),
          k(A.rbegin()[0]),
          n(B.rbegin()[0]),
          broadcaster({slice(A.data(), A.size() - 2),
                       slice(B.data(), B.size() - 2)}) {
    }
#undef A
#undef B

    dim_t MatMulIntegerInfo::batch() const noexcept {
        return broadcaster.outputsCount;
    }

}// namespace refactor::kernel
